import torch
from torch import nn
import torch.nn.functional as F


# helper function
def channel_shuffle(x, groups):
    b, n, h, w = x.shape
    channels_per_group = n // groups

    # reshape
    x = x.view(b, groups, channels_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(b, -1, h, w)

    return x


def basic_conv(in_channel, channel, kernel=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channel, channel, kernel, stride, kernel // 2, bias=False),
        nn.BatchNorm2d(channel), nn.ReLU(inplace=True)
    )


# basic module
# TODO: may add bn and relu
class DownSampling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownSampling, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, ceil_mode=True)
        self.bn = nn.BatchNorm2d(out_channel + in_channel)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = F.relu_(self.bn(x))
        return x


class SSnbt(nn.Module):
    def __init__(self, channel, dilate=1, drop_prob=0.01):
        super(SSnbt, self).__init__()
        channel = channel // 2
        self.left = nn.Sequential(
            nn.Conv2d(channel, channel, (3, 1), (1, 1), (1, 0)), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (1, 3), (1, 1), (0, 1), bias=False),
            nn.BatchNorm2d(channel), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (3, 1), (1, 1), (dilate, 0), dilation=(dilate, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (1, 3), (1, 1), (0, dilate), dilation=(1, dilate), bias=False),
            nn.BatchNorm2d(channel), nn.Dropout2d(drop_prob, inplace=True)
        )
        self.right = nn.Sequential(
            nn.Conv2d(channel, channel, (1, 3), (1, 1), (0, 1)), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (3, 1), (1, 1), (1, 0), bias=False),
            nn.BatchNorm2d(channel), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (1, 3), (1, 1), (0, dilate), dilation=(1, dilate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (3, 1), (1, 1), (dilate, 0), dilation=(dilate, 1), bias=False),
            nn.BatchNorm2d(channel), nn.Dropout2d(drop_prob, inplace=True)
        )

    def forward(self, x):
        x1, x2 = x.split(x.shape[1] // 2, 1)
        x1 = self.left(x1)
        x2 = self.right(x2)
        out = torch.cat([x1, x2], 1)
        x = F.relu(out + x)
        return channel_shuffle(x, 2)


class SSnbtv2(nn.Module):
    def __init__(self, channel, dilate=1, drop_prob=0.01):
        super(SSnbtv2, self).__init__()
        channel = channel // 2
        self.left = nn.Sequential(
            nn.Conv2d(channel, channel, (3, 1), (1, 1), (1, 0)), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (1, 3), (1, 1), (0, 1), bias=False),
            nn.BatchNorm2d(channel), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (3, 1), (1, 1), (dilate, 0), dilation=(dilate, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (1, 3), (1, 1), (0, dilate), dilation=(1, dilate), bias=False),
            nn.BatchNorm2d(channel), nn.ReLU(inplace=True), nn.Dropout2d(drop_prob, inplace=True)
        )
        self.right = nn.Sequential(
            nn.Conv2d(channel, channel, (1, 3), (1, 1), (0, 1)), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (3, 1), (1, 1), (1, 0), bias=False),
            nn.BatchNorm2d(channel), nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (1, 3), (1, 1), (0, dilate), dilation=(1, dilate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, (3, 1), (1, 1), (dilate, 0), dilation=(dilate, 1), bias=False),
            nn.BatchNorm2d(channel), nn.ReLU(inplace=True), nn.Dropout2d(drop_prob, inplace=True)
        )

    def forward(self, x):
        x1, x2 = x.split(x.shape[1] // 2, 1)
        x1 = self.left(x1)
        x2 = self.right(x2)
        out = torch.cat([x1, x2], 1)
        x = F.relu(out + x)
        return channel_shuffle(x, 2)

#原始APN
class APN(nn.Module):
    def __init__(self, channel, classes):
        super(APN, self).__init__()
        self.conv1 = basic_conv(channel, channel, 3, 2)
        self.conv2 = basic_conv(channel, channel, 5, 2)
        self.conv3 = basic_conv(channel, channel, 7, 2)
        self.branch1 = basic_conv(channel, classes, 1, 1)
        self.branch2 = basic_conv(channel, classes, 1, 1)
        self.branch3 = basic_conv(channel, classes, 1, 1)
        self.branch4 = basic_conv(channel, classes, 1, 1)
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            basic_conv(channel, classes, 1, 1)
        )

    def forward(self, x):
        _, _, h, w = x.shape
        out3 = self.conv1(x)
        out2 = self.conv2(out3)
        out = self.branch1(self.conv3(out2))
        out = F.interpolate(out, size=((h + 3) // 4, (w + 3) // 4), mode='bilinear', align_corners=True)
        out = out + self.branch2(out2)
        out = F.interpolate(out, size=((h + 1) // 2, (w + 1) // 2), mode='bilinear', align_corners=True)
        out = out + self.branch3(out3)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        out = out * self.branch4(x)
        out = out + self.branch5(x)
        return out


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=6, dilation=6)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=12, dilation=12)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=18, dilation=18)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.conv6 = nn.Conv2d(out_channels * 5, out_channels, 1, 1)  # 添加的额外卷积层

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        out5 = self.avg_pool(x)
        out5 = self.conv5(out5)
        out5 = F.interpolate(out5, size=x.size()[2:], mode='bilinear', align_corners=True)  # 使用插值调整尺寸
        out = torch.cat([out1, out2, out3, out4, out5], dim=1)
        out = self.conv6(out)  # 使用额外的卷积层调整通道数
        return out




class LEDNet(nn.Module):
    def __init__(self, nclass, drop=0.1):
        super(LEDNet, self).__init__()
        self.encoder = nn.Sequential(
            DownSampling(3, 29), SSnbt(32, 1, 0.1 * drop), SSnbt(32, 1, 0.1 * drop), SSnbt(32, 1, 0.1 * drop),
            DownSampling(32, 32), SSnbt(64, 1, 0.1 * drop), SSnbt(64, 1, 0.1 * drop),
            DownSampling(64, 64), SSnbt(128, 1, drop), SSnbt(128, 2, drop), SSnbt(128, 5, drop),
            SSnbt(128, 9, drop), SSnbt(128, 2, drop), SSnbt(128, 5, drop), SSnbt(128, 9, drop), SSnbt(128, 17, drop)
        )
        self.aspp = ASPP(128,128)
        #self.dense_aspp = _DenseASPPBlock(128, 32, 128)
        self.decoder = APN(128, nclass)

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.encoder(x)
        x = self.aspp(x)
        x = F.interpolate(x, size=((h + 3) // 4, (w + 3) // 4), mode='bilinear', align_corners=True)
        x = F.interpolate(x, size=((h + 1) // 2, (w + 1) // 2), mode='bilinear', align_corners=True)
        #x = self.dense_aspp(x)
        x = self.decoder(x)
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)


# if __name__ == '__main__':
#     net = LEDNet(21)
#     import torch
#
#     a = torch.randn(2, 3, 554, 253)
#     out = net(a)
#     print(out.shape)
#
# if __name__ == '__main__':
#     # model = DownSampling(32)
#     # a = torch.randn(1, 32, 512, 256)
#     # out = model(a)
#     # print(out.shape)
#
#     # model = SSnbt(10, 2)
#     # a = torch.randn(1, 20, 10, 10)
#     # out = model(a)
#     # print(out.shape)
#     # model = basic_conv(10, 20, 3, 2)
#     # a = torch.randn(1, 10, 128, 65)
#     # out = model(a)
#     # print(out.shape)
#
#     model = APN(64, 10)
#     x = torch.randn(2, 64, 127, 65)
#     out = model(x)
#     print(out.shape)