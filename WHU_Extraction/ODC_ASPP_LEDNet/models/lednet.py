from torch import nn
import torch.nn.functional as F
from models.basic import DownSampling, SSnbt, APN


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


# 计算模型的参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# 创建模型实例
model = LEDNet(nclass=2)

# 打印模型的参数量
params = count_parameters(model)
print(f"模型的参数量：{params/1e6}M")
if __name__ == '__main__':
    net = LEDNet(21)
    a = torch.randn(2, 3, 256, 256)
    out = net(a)
    print(out.shape)
