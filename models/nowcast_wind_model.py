# -*- coding: utf-8 -*-
# @Time    : 2025/10/15 10:43
# @Author  : wp@1122
# @File    : nowcast_wind_model.py
# @Desc    :
# nowcast_wind_model.py
"""
convnext_unet_convlstm.py

Nowcast-style model: ConvNeXt-like encoder + ConvLSTM temporal fusion + UNet decoder.
Inputs:
 - obs:  [B, 1, 3, 256, 256]
 - mode: [B, 2, 12, 256, 256]
 - static: [B, 1, 256, 256]  (optional)
Output:
 - pred: [B, 12, 256, 256]  (predicted windspeed per hour)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# -----------------------
# Basic building blocks
# -----------------------
class DepthwiseConv(nn.Module):
    def __init__(self, dim, kernel_size=7, padding=3):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim, bias=False)

    def forward(self, x):
        return self.dw(x)

class ConvNeXtBlockSimple(nn.Module):
    """
    Simplified ConvNeXt-like block:
      - depthwise conv
      - pointwise conv (linear expansion & projection)
      - GELU and layernorm style via BatchNorm fallback
    """
    def __init__(self, dim, expansion=4):
        super().__init__()
        hidden_dim = dim * expansion
        self.dw = DepthwiseConv(dim, kernel_size=7, padding=3)
        self.pw1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        residual = x
        x = self.dw(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.norm(x)
        return x + residual


class EncoderStage(nn.Module):
    def __init__(self, in_ch, out_ch, n_blocks=2):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(n_blocks):
            layers.append(ConvNeXtBlockSimple(out_ch))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DecoderStage(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # upsample by 2 then conv block
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            ConvNeXtBlockSimple(out_ch)
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # if shapes mismatch, interpolate skip
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


# -----------------------
# ConvLSTM cell (convolutional LSTM)
# -----------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=padding)

        self.hidden_ch = hidden_ch

    def forward(self, x, h, c):
        # x: [B, in_ch, H, W]
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)  # [B, 4*hidden_ch, H, W]
        i, f, o, g = torch.split(gates, self.hidden_ch, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, in_ch, hidden_ch, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            ic = in_ch if i == 0 else hidden_ch
            self.cells.append(ConvLSTMCell(ic, hidden_ch))

    def forward(self, x_seq, h0=None, c0=None):
        # x_seq: [B, T, C, H, W]
        B, T, C, H, W = x_seq.shape
        device = x_seq.device
        h = []
        c = []
        # init
        for layer in range(self.num_layers):
            if h0 is None:
                h.append(torch.zeros(B, self.cells[layer].hidden_ch, H, W, device=device))
                c.append(torch.zeros(B, self.cells[layer].hidden_ch, H, W, device=device))
            else:
                h.append(h0[layer])
                c.append(c0[layer])
        outputs = []
        for t in range(T):
            inp = x_seq[:, t]
            for layer in range(self.num_layers):
                cell = self.cells[layer]
                h_l, c_l = cell(inp, h[layer], c[layer])
                h[layer] = h_l
                c[layer] = c_l
                inp = h_l  # pass to next layer
            outputs.append(h_l.unsqueeze(1))
        # outputs: list of [B,1,hidden,H,W] -> cat
        outputs = torch.cat(outputs, dim=1)  # [B, T, hidden, H, W]
        # return all outputs and last states
        return outputs, (h, c)


# -----------------------
# Main Nowcast ConvNeXt-UNet-ConvLSTM model
# -----------------------
class ConvNextUNetConvLSTM(nn.Module):
    def __init__(self, base_ch=32, depth=4, lstm_hidden=64):
        """
        base_ch: channels at first encoder stage
        depth: number of downsampling stages (we'll do 4 by default)
        lstm_hidden: hidden channels for ConvLSTM at bottleneck
        """
        super().__init__()
        self.depth = depth
        # encoder stages
        chs = [base_ch * (2**i) for i in range(depth)]  # e.g. [32,64,128,256]
        self.enc_stages = nn.ModuleList()
        # first stage accepts varying input channels (we'll use dynamic conv before stage)
        for i in range(depth):
            in_ch = chs[i-1] if i > 0 else None
            # we'll create stage wrappers later
            stage = EncoderStage(chs[i-1] if i>0 else chs[i], chs[i]) if False else None
            # we will implement custom below to support dynamic in channels
            self.enc_stages.append(stage)
        # Instead of above confusion, explicitly define an encoder sequence that downsamples
        # We'll implement a simple encoder with downsample convs
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            ConvNeXtBlockSimple(base_ch),
            ConvNeXtBlockSimple(base_ch)
        )
        self.down1 = nn.Conv2d(base_ch, base_ch*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc2 = nn.Sequential(
            nn.BatchNorm2d(base_ch*2),
            nn.ReLU(inplace=True),
            ConvNeXtBlockSimple(base_ch*2),
            ConvNeXtBlockSimple(base_ch*2)
        )
        self.down2 = nn.Conv2d(base_ch*2, base_ch*4, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc3 = nn.Sequential(
            nn.BatchNorm2d(base_ch*4),
            nn.ReLU(inplace=True),
            ConvNeXtBlockSimple(base_ch*4),
            ConvNeXtBlockSimple(base_ch*4)
        )
        self.down3 = nn.Conv2d(base_ch*4, base_ch*8, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc4 = nn.Sequential(
            nn.BatchNorm2d(base_ch*8),
            nn.ReLU(inplace=True),
            ConvNeXtBlockSimple(base_ch*8),
            ConvNeXtBlockSimple(base_ch*8)
        )
        # Bottleneck conv to unify channels
        self.bottleneck_proj = nn.Conv2d(base_ch*8, lstm_hidden, kernel_size=1)

        # ConvLSTM for mode (12 steps) and obs (3 steps)
        self.mode_lstm = ConvLSTM(in_ch=lstm_hidden, hidden_ch=lstm_hidden, num_layers=1)
        self.obs_lstm = ConvLSTM(in_ch=lstm_hidden, hidden_ch=lstm_hidden, num_layers=1)

        # static encoder
        self.static_encoder = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_ch, lstm_hidden, 1)
        )

        # decoder stages (we will decode per time step; use skip connections from mode encoder)
        # decoder at lower resolution expects concatenated channels: hidden + skip_ch
        self.dec3 = DecoderStage(in_ch=lstm_hidden + base_ch*8, out_ch=base_ch*4)
        self.dec2 = DecoderStage(in_ch=base_ch*4 + base_ch*4, out_ch=base_ch*2)
        self.dec1 = DecoderStage(in_ch=base_ch*2 + base_ch*2, out_ch=base_ch)
        # final conv to single channel
        self.final_conv = nn.Conv2d(base_ch, 1, kernel_size=1)

    def encode_frame(self, x):
        """
        encode a single 2D frame x: [B, C=1 or 2, H, W]
        returns encoder features for 3 levels for skip connections and bottleneck feature:
          feat1: [B, base, H, W]
          feat2: [B, base*2, H/2, W/2]
          feat3: [B, base*4, H/4, W/4]
          feat4: [B, base*8, H/8, W/8]  (bottleneck pre-proj)
        """
        # If x has >1 channel, project to 1 channel first for enc1 conv (simple)
        if x.shape[1] != 1 and x.shape[1] != None:
            # project channels to 1 via conv
            x = nn.Conv2d(x.shape[1], 1, kernel_size=1).to(x.device)(x)
        f1 = self.enc1(x)            # [B, base, H, W]
        f2 = self.enc2(self.down1(f1))  # [B, base*2, H/2, W/2]
        f3 = self.enc3(self.down2(f2))  # [B, base*4, H/4, W/4]
        f4 = self.enc4(self.down3(f3))  # [B, base*8, H/8, W/8]
        return f1, f2, f3, f4

    def forward(self, obs, mode, static=None):
        """
        obs: [B, 1, 3, H, W]
        mode: [B, 2, 12, H, W]
        static: [B, 1, H, W] or None
        returns: preds [B, 12, H, W]
        """
        B = obs.shape[0]
        device = obs.device
        _, _, T_obs, H, W = obs.shape
        _, _, T_mode, _, _ = mode.shape

        # --- Encode frames per time for mode and obs. We will use mode encoder skips per time for UNet decoding
        # mode: list of per-time skip features
        mode_f1_list, mode_f2_list, mode_f3_list, mode_f4_list = [], [], [], []
        for t in range(T_mode):
            # mode[:, :, t, ...] shape => [B, 2, H, W]
            frame = mode[:, :, t, :, :]
            f1, f2, f3, f4 = self.encode_frame(frame)
            mode_f1_list.append(f1)
            mode_f2_list.append(f2)
            mode_f3_list.append(f3)
            mode_f4_list.append(f4)

        # obs encode: produce bottleneck seq for obs (3 steps)
        obs_f4_list = []
        for t in range(T_obs):
            frame = obs[:, :, t, :, :]  # [B,1,H,W]
            _, _, _, f4 = self.encode_frame(frame)
            obs_f4_list.append(f4)
        # project bottleneck features to lstm_in channels
        # stack to seq: [B, T, C, h, w]
        mode_bottleneck_seq = torch.stack([self.bottleneck_proj(f4) for f4 in mode_f4_list], dim=1)  # [B, T_mode, lstm_hidden, h8, w8]
        obs_bottleneck_seq = torch.stack([self.bottleneck_proj(f4) for f4 in obs_f4_list], dim=1)     # [B, T_obs, lstm_hidden, h8, w8]

        # static encoding
        if static is not None:
            static_feat = self.static_encoder(static)  # [B, lstm_hidden, H, W]
            # downsample static to h8,w8
            static_feat8 = F.interpolate(static_feat, size=mode_bottleneck_seq.shape[-2:], mode='bilinear', align_corners=False)
        else:
            static_feat8 = torch.zeros(B, mode_bottleneck_seq.shape[2], mode_bottleneck_seq.shape[3], mode_bottleneck_seq.shape[4], device=device)

        # --- ConvLSTM temporal encoding
        mode_outputs, _ = self.mode_lstm(mode_bottleneck_seq)  # [B, T_mode, hidden, h8, w8]
        obs_outputs, (obs_states, _) = self.obs_lstm(obs_bottleneck_seq)    # [B, T_obs, hidden, h8, w8] ; obs_states = [h] list
        # get last obs hidden state (layer 0)
        obs_h_last = obs_outputs[:, -1]  # [B, hidden, h8, w8]

        # --- For each future step t in 0..T_mode-1, fuse mode_outputs[:,t] with obs_h_last and static
        preds = []
        for t in range(T_mode):
            m_t = mode_outputs[:, t]  # [B, hidden, h8, w8]
            # fuse by concat along channel dim
            fused = torch.cat([m_t, obs_h_last, static_feat8], dim=1)  # channel = hidden*2 + hidden = hidden*3
            # reduce to bottleneck channel
            # project fused to hidden size
            fused_proj = nn.Conv2d(fused.shape[1], self.bottleneck_proj.out_channels, kernel_size=1).to(device)(fused)
            # decoder: use mode skip features at time t
            skip3 = mode_f3_list[t]  # [B, base*4, h4,w4]
            skip2 = mode_f2_list[t]
            skip1 = mode_f1_list[t]
            # upsample fused_proj to decoder resolution and decode
            x = fused_proj  # [B, hidden, h8, w8]
            # dec3 expects in_ch = hidden + base*8  => we concat fused_proj with skip4 (mode_f4)
            # But easier: we will upsample to h4,w4 and concat skip3
            x = F.interpolate(x, size=skip3.shape[-2:], mode='bilinear', align_corners=False)  # [B, hidden, h4,w4]
            x = torch.cat([x, skip3], dim=1)  # channel = hidden + base*4
            x = self.dec3.conv(x)  # using decoder conv path
            # dec2
            x = self.dec2.up(x)
            if skip2.shape[-2:] != x.shape[-2:]:
                skip2 = F.interpolate(skip2, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip2], dim=1)
            x = self.dec2.conv(x)
            # dec1
            x = self.dec1.up(x)
            if skip1.shape[-2:] != x.shape[-2:]:
                skip1 = F.interpolate(skip1, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip1], dim=1)
            x = self.dec1.conv(x)
            # final
            out = self.final_conv(x)  # [B,1,H,W]
            preds.append(out.unsqueeze(1))  # [B,1,1,H,W]
        preds = torch.cat(preds, dim=1)  # [B, T_mode, 1, H, W]
        preds = preds.squeeze(2)  # [B, T_mode, H, W]
        return preds


# -----------------------
# Quick smoke test
# -----------------------
if __name__ == "__main__":
    B = 2
    H = W = 256
    obs = torch.randn(B, 1, 3, H, W)
    mode = torch.randn(B, 2, 12, H, W)
    static = torch.randn(B, 1, H, W)
    model = ConvNextUNetConvLSTM(base_ch=24, depth=4, lstm_hidden=64)
    preds = model(obs, mode, static)  # [B,12,H,W]
    print("preds shape:", preds.shape)

"""
设计思路：
单个样本记录(10月15日08时)如下：
1、输入：
    1）过去3个小时(6时、7时、8时)的实况雷暴大风:[1, 3, 256, 256]；
    2) 05时起报(004时效-015时效)未来12小时的组合反射率、阵风预报:[2, 12, 256, 256]；
    3) 后续考虑加入静态地形数据:[256, 256]；
2、输出：
    未来12小时(10月15日09时-20时)雷暴大风风速（实况预估）
"""