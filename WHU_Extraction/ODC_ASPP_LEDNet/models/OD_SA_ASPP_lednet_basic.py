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

#
# class ASPP(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ASPP, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1)
#         self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=6, dilation=6)
#         self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=12, dilation=12)
#         self.conv4 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=18, dilation=18)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv5 = nn.Conv2d(in_channels, out_channels, 1, 1)
#         self.conv6 = nn.Conv2d(out_channels * 5, out_channels, 1, 1)  # 添加的额外卷积层
#
#     def forward(self, x):
#         out1 = self.conv1(x)
#         out2 = self.conv2(x)
#         out3 = self.conv3(x)
#         out4 = self.conv4(x)
#         out5 = self.avg_pool(x)
#         out5 = self.conv5(out5)
#         out5 = F.interpolate(out5, size=x.size()[2:], mode='bilinear', align_corners=True)  # 使用插值调整尺寸
#         out = torch.cat([out1, out2, out3, out4, out5], dim=1)
#         out = self.conv6(out)  # 使用额外的卷积层调整通道数
#         return out

# class ASPP(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ASPP, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1)
#         self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=6, dilation=6)
#         self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=12, dilation=12)
#         self.conv4 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=18, dilation=18)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv5 = nn.Conv2d(in_channels, out_channels, 1, 1)
#         self.conv6 = nn.Conv2d(out_channels * 5, out_channels, 1, 1)  # 添加的额外卷积层
#         self.self_attention = SelfAttention(out_channels)  # 添加的自注意力机制模块
#
#     def forward(self, x):
#         out1 = self.conv1(x)
#         out2 = self.conv2(x)
#         out3 = self.conv3(x)
#         out4 = self.conv4(x)
#         out5 = self.avg_pool(x)
#         out5 = self.conv5(out5)
#         out5 = F.interpolate(out5, size=x.size()[2:], mode='bilinear', align_corners=True)  # 使用插值调整尺寸
#         out = torch.cat([out1, out2, out3, out4, out5], dim=1)
#         out = self.conv6(out)  # 使用额外的卷积层调整通道数
#         # 自注意力机制
#         attention_weights = self.self_attention(out)
#         weighted_features = out * attention_weights
#
#         out = weighted_features
#
#         return out

# class ASPP(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ASPP, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1)
#         self.sa1 = SelfAttention(out_channels)  # 添加自注意力模块
#         self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=6, dilation=6)
#         self.sa2 = SelfAttention(out_channels)  # 添加自注意力模块
#         self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=12, dilation=12)
#         self.sa3 = SelfAttention(out_channels)  # 添加自注意力模块
#         self.conv4 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=18, dilation=18)
#         self.sa4 = SelfAttention(out_channels)  # 添加自注意力模块
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv5 = nn.Conv2d(in_channels, out_channels, 1, 1)
#         self.conv6 = nn.Conv2d(out_channels * 5, out_channels, 1, 1)
#
#     def forward(self, x):
#         out1 = self.conv1(x)
#         out1 = self.sa1(out1)
#         out2 = self.conv2(x)
#         out2 = self.sa2(out2)
#         out3 = self.conv3(x)
#         out3 = self.sa3(out3)
#         out4 = self.conv4(x)
#         out4 = self.sa4(out4)
#         out5 = self.avg_pool(x)
#         out5 = self.conv5(out5)
#         out5 = F.interpolate(out5, size=x.size()[2:], mode='bilinear', align_corners=True)
#         out = torch.cat([out1, out2, out3, out4, out5], dim=1)
#         out = self.conv6(out)
#         return out


class ASPP(nn.Module):#可以运行
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.attention_blocks = nn.ModuleList([
            SelfAttention(out_channels) for _ in range(5)  # 为每个路径创建独立的注意力模块
        ])
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=6, dilation=6)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=12, dilation=12)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=18, dilation=18)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.conv6 = nn.Conv2d(out_channels * 5, out_channels, 1, 1)
        self.bottleneck = nn.Conv2d(out_channels, out_channels, 1)  # 通道数量需要根据实际修改

    def forward(self, x):
        features = []
        conv_outputs = [
            self.conv1(x),
            self.conv2(x),
            self.conv3(x),
            self.conv4(x),
            F.interpolate(self.conv5(self.avg_pool(x)), size=x.size()[2:], mode='bilinear', align_corners=True)
        ]

        # 对每个卷积输出应用相应的自注意力机制
        for idx, conv_output in enumerate(conv_outputs):
            features.append(self.attention_blocks[idx](conv_output))

        # 将加权特征进行合并
        out = torch.sum(torch.stack(features, dim=0), dim=0)  # 沿新建的维度求和，实现特征加权融合

        # 经过一个收缩层，将输出通道数调整到所需的输出通道数
        output = self.bottleneck(out)
        return output

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # B x N x C
        proj_key = self.key(x).view(batch_size, -1, width * height)  # B x C x N
        energy = torch.bmm(proj_query, proj_key)  # B x N x N
        attention = self.softmax(energy)  # B x N x N
        proj_value = self.value(x).view(batch_size, -1, width * height)  # B x C x N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x N
        out = out.view(batch_size, C, width, height)

        return out
# class APN(nn.Module):
#     def __init__(self, in_channels, nclass):
#         super(APN, self).__init__()
#         self.branch1 = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, 3, padding=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, nclass, 1)
#         )
#         self.branch2 = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, 3, padding=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, nclass, 1)
#         )
#         self.branch3 = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, 3, padding=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, nclass, 1)
#         )
#         self.branch4 = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, 3, padding=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, nclass, 1)
#         )
#         self.branch5 = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, 3, padding=1),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, nclass, 1)
#         )
#
#     def forward(self, x):
#         _, _, h, w = x.shape
#         out3 = self.conv1(x)
#         out2 = self.conv2(out3)
#         out = self.branch1(self.conv3(out2))
#         out = F.interpolate(out, size=((h + 3) // 4, (w + 3) // 4), mode='bilinear', align_corners=True)
#         out = out + self.branch2(out2)
#         out = F.interpolate(out, size=((h + 1) // 2, (w + 1) // 2), mode='bilinear', align_corners=True)
#         out = out + self.branch3(out3)
#         out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
#         out = out * self.branch4(x)
#         out = out + self.branch5(x)
#         return out
#
# class _DenseASPPConv(nn.Sequential):
#     def __init__(self, in_channels, inter_channels, out_channels, atrous_rate, drop_rate=0.1, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
#         super(_DenseASPPConv, self).__init__()
#         self.add_module('conv1', nn.Conv2d(in_channels, inter_channels, 1, bias=False))
#         self.add_module('bn1', norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)))
#         self.add_module('relu1', nn.ReLU(True))
#         self.add_module('conv2', nn.Conv2d(inter_channels, out_channels, 3, dilation=atrous_rate, padding=atrous_rate))
#         self.add_module('bn2', norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)))
#         self.add_module('relu2', nn.ReLU(True))
#         self.drop_rate = drop_rate
#
#     def forward(self, x):
#         features = super(_DenseASPPConv, self).forward(x)
#         if self.drop_rate > 0:
#             features = F.dropout(features, p=self.drop_rate, training=self.training)
#         return features
#
# class _DenseASPPBlock(nn.Module):
#     def __init__(self, in_channels, inter_channels1, inter_channels2, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
#         super(_DenseASPPBlock, self).__init__()
#         self.aspp_3 = _DenseASPPConv(in_channels, inter_channels1, inter_channels2, 3, 0.1, norm_layer, norm_kwargs)
#         self.aspp_6 = _DenseASPPConv(in_channels + inter_channels2 * 1, inter_channels1, inter_channels2, 6, 0.1, norm_layer, norm_kwargs)
#         self.aspp_12 = _DenseASPPConv(in_channels + inter_channels2 * 2, inter_channels1, inter_channels2, 12, 0.1, norm_layer, norm_kwargs)
#         self.aspp_18 = _DenseASPPConv(in_channels + inter_channels2 * 3, inter_channels1, inter_channels2, 18, 0.1, norm_layer, norm_kwargs)
#         self.aspp_24 = _DenseASPPConv(in_channels + inter_channels2 * 4, inter_channels1, inter_channels2, 24, 0.1, norm_layer, norm_kwargs)
#
#     def forward(self, x):
#         aspp3 = self.aspp_3(x)
#         x = torch.cat([aspp3, x], dim=1)
#         aspp6 = self.aspp_6(x)
#         x = torch.cat([aspp6, x], dim=1)
#         aspp12 = self.aspp_12(x)
#         x = torch.cat([aspp12, x], dim=1)
#         aspp18 = self.aspp_18(x)
#         x = torch.cat([aspp18, x], dim=1)
#         aspp24 = self.aspp_24(x)
#         x = torch.cat([aspp24, x], dim=1)
#         return x