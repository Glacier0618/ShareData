from torch import nn
import torch
from models.Unet.unet_parts import OutConv
from models.Unet.unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
from models.Unet.layers import CBAM

a = 4
class SmaAt_UNet(nn.Module):
    def __init__(
        self,
        input_channels,
        out_channels,
        kernels_per_layer=2,
        bilinear=True,
        reduction_ratio=16,
    ):
        super(SmaAt_UNet, self).__init__()
        self.input_channels = input_channels
        self.out_channels = out_channels
        kernels_per_layer = kernels_per_layer
        self.bilinear = bilinear
        reduction_ratio = reduction_ratio

        self.inc = DoubleConvDS(self.input_channels, 64//a, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64//a, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64//a, 128//a, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128//a, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128//a, 256//a, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256//a, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256//a, 512//a, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512//a, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512//a, 1024 // factor//a, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024 // factor//a, reduction_ratio=reduction_ratio)
        self.up1 = UpDS(1024//a, 512 // factor//a, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512//a, 256 // factor//a, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256//a, 128 // factor//a, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128//a, 64//a, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.outc = OutConv(64//a, self.out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)

        logits = self.outc(x)

        return logits

# if __name__=='__main__':
#     # input ([B, 24, 2, 567, 567])
#     # label ([B, 24, 433, 433]   )
#
#     loss_f = nn.CrossEntropyLoss()
#     input = torch.zeros(4, 48, 567, 567)
#     label = torch.round(torch.rand(4, 24, 433, 433))
#     model = SmaAt_UNet(input_channels=48, out_channels=24)
#     out = model(input)
#     loss =loss_f(out, label)
#     print(loss)

