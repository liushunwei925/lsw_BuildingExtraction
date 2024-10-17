import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.down1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)

        self.middle = DoubleConv(256, 512)

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv3 = DoubleConv(128, 64)
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.upconv4 = DoubleConv(64, 32)

        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):

        down1 = self.down1(x)
        pool1 = self.pool1(down1)
        down2 = self.down2(pool1)
        pool2 = self.pool2(down2)
        down3 = self.down3(pool2)
        pool3 = self.pool3(down3)
        down4 = self.down4(pool3)
        pool4 = self.pool4(down4)

        middle = self.middle(pool4)

        up1 = self.up1(middle)
        concat1 = torch.cat([down4, up1], dim=1)
        upconv1 = self.upconv1(concat1)

        up2 = self.up2(upconv1)
        concat2 = torch.cat([down3, up2], dim=1)
        upconv2 = self.upconv2(concat2)

        up3 = self.up3(upconv2)
        concat3 = torch.cat([down2, up3], dim=1)
        upconv3 = self.upconv3(concat3)

        up4 = self.up4(upconv3)
        concat4 = torch.cat([down1, up4], dim=1)
        upconv4 = self.upconv4(concat4)

        out = self.out_conv(upconv4)
        return out
# 创建UNet模型实例
model = UNet(3,2)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 打印UNet模型的参数量
params = count_parameters(model)
print(f"UNet模型的参数量：{params/1e6}M")