import torch
from torch import nn
import torch.nn.functional as F

#  加入ODConv
class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class ODConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)

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
# class DownSampling(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(DownSampling, self).__init__()
#         self.conv = nn.Conv2d(in_channel, out_channel, 3, stride=2, padding=1)
#         self.pool = nn.MaxPool2d(2, ceil_mode=True)
#         self.bn = nn.BatchNorm2d(out_channel + in_channel)
#
#     def forward(self, x):
#         x1 = self.conv(x)
#         x2 = self.pool(x)
#         x = torch.cat([x1, x2], dim=1)
#         x = F.relu_(self.bn(x))
#         return x
class DownSampling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownSampling, self).__init__()
        self.conv = ODConv2d(in_channel, out_channel, 3,stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, ceil_mode=True)
        self.bn = nn.BatchNorm2d(out_channel + in_channel)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.bn(x))

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

class LEDNet(nn.Module):
    def __init__(self, nclass, drop=0.1):
        super(LEDNet, self).__init__()
        self.encoder = nn.Sequential(
            DownSampling(3, 29), SSnbt(32, 1, 0.1 * drop), SSnbt(32, 1, 0.1 * drop), SSnbt(32, 1, 0.1 * drop),
            DownSampling(32, 32), SSnbt(64, 1, 0.1 * drop), SSnbt(64, 1, 0.1 * drop),
            DownSampling(64, 64), SSnbt(128, 1, drop), SSnbt(128, 2, drop), SSnbt(128, 5, drop),
            SSnbt(128, 9, drop), SSnbt(128, 2, drop), SSnbt(128, 5, drop), SSnbt(128, 9, drop), SSnbt(128, 17, drop)
        )
        self.decoder = APN(128, nclass)

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.encoder(x)
        x = self.decoder(x)
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)