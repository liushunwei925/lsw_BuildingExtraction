import torch.nn as nn
import torch


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        # 加载预训练的 ResNet-50 模型
        self.resnet = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)

        # 修改第一层卷积的输入通道数为3，以适应彩色图像
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 修改最后一层全连接层的输出类别数
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)