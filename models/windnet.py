# -*- coding: utf-8 -*-
# @Time    : 2025/10/16 16:37
# @Author  : wp@1122
# @File    : windnet.py
# @Desc    :
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LayerNorm2d(nn.Module):
    """2D Layer Normalization"""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: [B, C, H, W]
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)  # 使用2D LayerNorm
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1)  # 使用Conv2d而不是Linear
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1)
        self.gamma = nn.Parameter(layer_scale * torch.ones((dim)),
                                  requires_grad=True) if layer_scale > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)  # 直接应用2D LayerNorm
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma.unsqueeze(-1).unsqueeze(-1) * x
        x = input + x
        return x


class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        super().__init__()
        self.downsample_layers = nn.ModuleList()

        # Stem layer
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0])
        )
        self.downsample_layers.append(stem)

        # Downsample layers
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[
                ConvNeXtBlock(dims[i]) for _ in range(depths[i])
            ])
            self.stages.append(stage)

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x  # [B, C, H, W]


class ConvNeXt3D(nn.Module):
    """
    3D version of ConvNeXt model, modified from the original 2D design:cite[1]:cite[9]:cite[10].
    Key changes include using 3D convolutions and 3D adaptations of the core components.
    """

    def __init__(self, in_chans=3, depths=[2, 2, 4, 2], dims=[64, 128, 256, 512]):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()

        # Initial stem with 3D convolution for patchifying the input
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv3d(in_chans, dims[0], kernel_size=(1, 4, 4), stride=(1, 4, 4), padding=(0, 0, 0)),
                nn.BatchNorm3d(dims[0]),
                nn.GELU()
            )
        )

        # Create stages with 3D ConvNeXt blocks
        for i in range(4):
            stage = nn.Sequential(*[
                ConvNeXt3DBlock(dims[i]) for _ in range(depths[i])
            ])
            self.stages.append(stage)

            if i < 3:
                self.downsample_layers.append(
                    nn.Sequential(
                        nn.BatchNorm3d(dims[i]),
                        nn.Conv3d(dims[i], dims[i + 1], kernel_size=(1, 2, 2), stride=(1, 2, 2)),
                    )
                )

    def forward(self, x):
        # x shape: [B, C, T, H, W] - assuming T is temporal dimension
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x


class ConvNeXt3DBlock(nn.Module):
    """3D version of ConvNeXt block with depthwise convolution"""

    def __init__(self, dim):
        super().__init__()

        # 3D depthwise convolution with larger kernel in spatial dimensions
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=(1, 7, 7), padding=(0, 3, 3), groups=dim)
        self.norm = nn.LayerNorm(dim)

        # Pointwise/1x1 convolutions for channel mixing
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        input = x
        x = self.dwconv(x)

        # Permute for LayerNorm and Linear layers
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)  # [B, T, H, W, C]
        x = self.norm(x)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        x = x.permute(0, 4, 1, 2, 3)  # [B, C, T, H, W]
        x = input + x
        return x


class UltraFastSpatialEncoder(nn.Module):
    def __init__(self, in_chans=3, out_chans=256):
        super().__init__()

        self.encoder = nn.Sequential(
            # 快速下采样 + 特征提取
            nn.Conv2d(in_chans, 64, 4, stride=2, padding=1),  # 256->128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 128->64
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 64->32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 全局平均池化 + 上采样（大幅减少计算量）
            nn.AdaptiveAvgPool2d(16),  # 32->16
            nn.Conv2d(256, 256, 1),  # 通道调整
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: [B, 3, 256, 256]
        x = self.encoder(x)  # [B, 256, 16, 16]
        x = F.interpolate(x, size=256, mode='bilinear')  # [B, 256, 256, 256]
        return x


class ConvLSTMCell(nn.Module):
    """ConvLSTM单元"""

    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        self.conv_ih = nn.Conv2d(input_dim, 4 * hidden_dim, kernel_size, padding=padding)
        self.conv_hh = nn.Conv2d(hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)

    def forward(self, x, hidden_state):
        h_prev, c_prev = hidden_state

        gates_input = self.conv_ih(x)
        gates_hidden = self.conv_hh(h_prev)
        gates = gates_input + gates_hidden

        i_gate, f_gate, g_gate, o_gate = torch.split(gates, self.hidden_dim, dim=1)

        i_t = torch.sigmoid(i_gate)
        f_t = torch.sigmoid(f_gate)
        g_t = torch.tanh(g_gate)
        o_t = torch.sigmoid(o_gate)

        c_cur = f_t * c_prev + i_t * g_t
        h_cur = o_t * torch.tanh(c_cur)

        return h_cur, c_cur

class ConvLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size=3, num_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(ConvLSTMCell(layer_input_dim, hidden_dim, kernel_size))

    def forward(self, x, hidden_state=None):
        # x: [B, T, C, H, W]
        b, t, c, h, w = x.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(b, h, w, x.device)

        current_states = list(hidden_state)
        layer_output = x

        output_sequence = []

        for time_step in range(t):
            input_frame = layer_output[:, time_step]

            new_states = []
            for layer_idx, cell in enumerate(self.cells):
                h_cur, c_cur = cell(input_frame, current_states[layer_idx])
                new_states.append((h_cur, c_cur))
                input_frame = h_cur

            current_states = new_states
            output_sequence.append(input_frame.unsqueeze(1))

        return torch.cat(output_sequence, dim=1), current_states

    def _init_hidden(self, batch_size, height, width, device):
        hidden = []
        for _ in range(self.num_layers):
            h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
            c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
            hidden.append((h, c))
        return hidden


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits



class ThunderstormWindNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.config = config
        # self.use_past_wind = config.get('use_past_wind', True)
        # self.use_terrain = config.get('use_terrain', True)
        # print(f"参数设置:{config}")
        # 空间特征提取分支（过去3小时实况）
        # if self.use_past_wind:
        # self.spatial_encoder = ConvNeXt(
        #     in_chans=3,
        #     depths=[2, 2, 4, 2],
        #     dims=[32, 64, 128, 256]
        # )
        self.spatial_encoder = UltraFastSpatialEncoder(in_chans=3, out_chans=256)

        self.spatial_proj = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # 时序特征提取分支（未来12小时预报）
        self.temporal_encoder = ConvLSTM(
            input_dim=2,  # 每个时间步有2个通道
            hidden_dim=64,
            kernel_size=3,
            num_layers=1
        )
        self.temporal_proj = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # 地形特征提取
        # if self.use_terrain:
        #     self.terrain_encoder = nn.Sequential(
        #         nn.Conv2d(1, 64, 3, padding=1),
        #         nn.BatchNorm2d(64),
        #         nn.ReLU(),
        #         nn.Conv2d(64, 128, 3, padding=1),
        #         nn.BatchNorm2d(128),
        #         nn.ReLU(),
        #         nn.Conv2d(128, 256, 3, padding=1),
        #         nn.BatchNorm2d(256),
        #         nn.ReLU()
        #     )

        # 计算融合后的通道数
        fusion_channels = 128  # 时序特征
        # if self.use_past_wind:
        fusion_channels += 128

        # if self.use_terrain:
        #     fusion_channels += 256

        # print(f"融合通道数: {fusion_channels}")

        # UNet作为融合和预测网络
        self.unet = UNet(
            n_channels=fusion_channels,
            n_classes=12,  # 输出未来12小时
            bilinear=True
        )

    def forward(self, past_wind, future_forecast) -> torch.Tensor:
                # terrain: Optional[torch.Tensor] = None) -> torch.Tensor:

        features = []
        # 过去实况特征提取
        # if self.use_past_wind and past_wind is not None:
        # print(f"过去实况输入: {past_wind.shape}")
        spatial_feat = self.spatial_encoder(past_wind)
        # print(f"空间特征: {spatial_feat.shape}")
        spatial_feat = self.spatial_proj(spatial_feat)
        spatial_feat = F.interpolate(spatial_feat, size=256, mode='bilinear')
        # print(f"空间特征上采样后: {spatial_feat.shape}")
        features.append(spatial_feat)

        # 未来预报时序特征提取
        # if future_forecast is not None:
        # print(f"未来预报输入: {future_forecast.shape}")
        B, C, T, H, W = future_forecast.shape
        # 调整维度为 [B, T, C, H, W]
        temporal_input = future_forecast.permute(0, 2, 1, 3, 4)
        # print(f"时序输入调整后: {temporal_input.shape}")
        temporal_feat, _ = self.temporal_encoder(temporal_input)
        # print(f"时序特征: {temporal_feat.shape}")
        # 取最后一个时间步的特征
        temporal_feat = temporal_feat[:, -1]
        temporal_feat = self.temporal_proj(temporal_feat)
        # print(f"时序特征投影后: {temporal_feat.shape}")
        features.append(temporal_feat)

        # 地形特征提取
        # if self.use_terrain and terrain is not None:
        #     # print(f"地形输入: {terrain.shape}")
        #     terrain_feat = self.terrain_encoder(terrain.unsqueeze(1))
        #     # print(f"地形特征: {terrain_feat.shape}")
        #     features.append(terrain_feat)

        # print(f"特征列表长度: {len(features)}")
        # for i, feat in enumerate(features):
        #     print(f"特征{i}: {feat.shape}")

        fused_feat = torch.cat(features, dim=1)
        # print(f"融合特征: {fused_feat.shape}")

        # UNet预测
        output = self.unet(fused_feat)
        # print(f"最终输出: {output.shape}")

        return output


# if __name__ == '__main__':
#     config = {
#         'use_past_wind': True,  # 是否使用过去3小时实况
#         'use_terrain': False,  # 是否使用地形数据
#     }
#     B, H, W = 2, 256, 256
#     # 初始化模型
#     model = ThunderstormWindNet()
#     future_forecast = torch.randn(B, 2, 12, H, W)
#     past_wind = torch.randn(B, 3, H, W)
#     # terrain = torch.randn(B, H, W)
#
#     # 完整输入
#     # output1 = model(past_wind, future_forecast, terrain)
#     # print(output1.shape)
#     # 实况+预报
#     output2 = model(past_wind, future_forecast)
#     print(output2.shape)
#     # 预报+地形
#     # output3 = model(None, future_forecast, terrain)
#     # print(output3.shape)
#     # 预报
#     # output4 = model(None, future_forecast, None)
#     # print(output4.shape)
