# -*- coding: utf-8 -*-
# @Time    : 2025/10/22 11:37
# @Author  : wp@1122
# @File    : temporal_unet.py
# @Desc    :

from typing import Tuple, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ThunderstormWindNet3D",
    "Fast3DEncoder",
    "LiteFPN",
]

# -------------------------------
# 3D 轻量模块
# -------------------------------
class DWConv3d(nn.Module):
    def __init__(self, c: int, k=(3, 3, 3), s=(1, 2, 2), p=(1, 1, 1)):
        super().__init__()
        self.dw = nn.Conv3d(c, c, kernel_size=k, stride=s, padding=p, groups=c, bias=False)
        self.pw = nn.Conv3d(c, c, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm3d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)

class SE3d(nn.Module):
    def __init__(self, c: int, r: int = 8):
        super().__init__()
        self.seq = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(c, max(1, c // r), 1), nn.SiLU(),
            nn.Conv3d(max(1, c // r), c, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.seq(x)

class Fast3DStage(nn.Module):
    """两层 DW3D block + 可选 SE；默认仅在空间降采样（stride=(1,2,2)）。"""
    def __init__(self, c: int, use_se: bool = True, down_spatial: bool = True):
        super().__init__()
        s = (1, 2, 2) if down_spatial else (1, 1, 1)
        self.b1 = DWConv3d(c, s=s)
        self.b2 = DWConv3d(c, s=(1, 1, 1))
        self.se = SE3d(c) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.b1(x)
        x = self.b2(x)
        x = self.se(x)
        return x

class Fast3DEncoder(nn.Module):
    """
    输入: [B, C, T, H, W]; 仅空间下采样 2~3 次 (总↓8)，保持或轻降时间维；
    输出: 2D 特征 [B, 2*base, H/8, W/8]
    """
    def __init__(self, in_c: int, base: int = 32, use_se: bool = True, reduce_time: bool = False):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_c, base, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(base), nn.SiLU(inplace=True)
        )  # -> H/2, W/2
        self.s1 = Fast3DStage(base, use_se=use_se, down_spatial=True)   # -> H/4, W/4
        self.s2 = Fast3DStage(base, use_se=use_se, down_spatial=True)   # -> H/8, W/8
        self.s3 = Fast3DStage(base, use_se=use_se, down_spatial=False)  # 保持 H/8, W/8

        self.reduce_time = reduce_time
        if reduce_time:
            # 轻降时间维（stride=2），仅用于 NWP 分支
            self.t_pool = nn.Conv3d(base, base, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))

        self.t_fuse = nn.Conv3d(base, base, kernel_size=1)   # 1x1x1 融合时间信息
        self.proj2d = nn.Conv2d(base, base * 2, kernel_size=3, padding=1)  # 2D 投影，提高通道

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        x = self.stem(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        if self.reduce_time:
            x = self.t_pool(x)
        x = self.t_fuse(x)           # [B, base, T', H/8, W/8]
        x = x.mean(dim=2)            # 时间平均 -> [B, base, H/8, W/8]
        x = self.proj2d(x)           # [B, 2*base, H/8, W/8]
        return x

# -------------------------------
# 2D 轻解码（可替换为更完整的 UNet-lite/FPN）
# -------------------------------
class LiteFPN(nn.Module):
    def __init__(self, in_c: int, out_hours: int = 12, mid: int = 64):
        super().__init__()
        self.lateral = nn.Conv2d(in_c, mid, 1)
        self.out = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1), nn.BatchNorm2d(mid), nn.SiLU(True),
            nn.Conv2d(mid, out_hours, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lateral(x)
        # 三次 2× 上采样，H/8 -> H
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.out(x)  # [B, 12, H, W]

# -------------------------------
# 主网络
# -------------------------------
class ThunderstormWindNet3D(nn.Module):
    """
    obs: [B, 3, H, W]       -> 视作 [B, 1, 3, H, W]
    nwp: [B, Cnwp, 12, H, W] (Cnwp=2 或 3)
    输出：
      y_reg: [B, 12, H, W]
      y_cls: [B, 12*K, H, W]  (K=len(thresholds))
    """
    def __init__(
        self,
        base_channels: int = 32,
        thresholds: Sequence[float] = (10.0, 15.0, 20.0, 25.0),
        use_topo: bool = False,
        use_se: bool = True,
        nwp_reduce_time: bool = True,
        out_hours: int = 12,
    ):
        super().__init__()
        self.out_hours = out_hours
        self.thresholds = list(thresholds)
        self.num_thr = len(self.thresholds)

        # 3D 编码器
        self.enc_obs = Fast3DEncoder(in_c=1, base=base_channels, use_se=use_se, reduce_time=False)
        in_c_nwp = 2 + (1 if use_topo else 0)
        self.enc_nwp = Fast3DEncoder(in_c=in_c_nwp, base=base_channels, use_se=use_se, reduce_time=nwp_reduce_time)

        # 融合 & 头部
        # enc_obs -> 2*base, enc_nwp -> 2*base  => 融合后 4*base -> 2*base
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * base_channels, 2 * base_channels, 1, bias=False),
            nn.BatchNorm2d(2 * base_channels), nn.SiLU(True)
        )
        self.reg_head = LiteFPN(in_c=2 * base_channels, out_hours=out_hours, mid=64)
        self.cls_head = nn.Sequential(
            nn.Conv2d(2 * base_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.SiLU(True),
            nn.Conv2d(64, out_hours * self.num_thr, 1)
        )

        # 小初始化：有助于稳定分类头早期训练
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, past_wind: torch.Tensor, future_forecast: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        past_wind: [B, 3, H, W]
        future_forecast: [B, Cnwp, 12, H, W]
        """
        # 观测 3 小时 -> [B,1,3,H,W]
        x_obs = past_wind.unsqueeze(1)
        f_obs = self.enc_obs(x_obs)           # [B, 2*base, H/8, W/8]

        # NWP 12 步 -> [B,Cnwp,12,H,W]
        f_nwp = self.enc_nwp(future_forecast) # [B, 2*base, H/8, W/8]

        f = torch.cat([f_obs, f_nwp], dim=1)  # [B, 4*base, H/8, W/8]
        f = self.fuse(f)                      # [B, 2*base, H/8, W/8]

        y_reg = self.reg_head(f)              # [B, 12, H, W]
        # 分类 head 直接在 H,W 分辨率上输出 12*K 通道 logits
        y_cls = self.cls_head(
            F.interpolate(f, scale_factor=2, mode='bilinear', align_corners=False) # -> H/4
        )
        y_cls = F.interpolate(y_cls, scale_factor=2, mode='bilinear', align_corners=False) # -> H/2
        y_cls = F.interpolate(y_cls, scale_factor=2, mode='bilinear', align_corners=False) # -> H

        return y_reg, y_cls


# -------------------------------
# 简易自测
# # -------------------------------
# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     B, H, W = 2, 256, 256
#     model = ThunderstormWindNet3D(base_channels=32, thresholds=(10,15,20,25), use_topo=False).to(device)
#     past = torch.randn(B, 3, H, W, device=device)
#     nwp = torch.randn(B, 2, 12, H, W, device=device)
#     y_reg, y_cls = model(past, nwp)
#     print("y_reg:", y_reg.shape, "y_cls:", y_cls.shape)
