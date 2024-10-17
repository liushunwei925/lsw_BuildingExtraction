from config import *
from models import deeplabv3_plus
from models import UNet
import torch
import torchvision.models as models
import torch.nn as nn
from models import VGGNet
from models import SegNet
from models import ResNet
from models import CLEDNet
from models import LEDNet2
from models import CMLEDNet
from models import LEDNet2
from models import CBAMLEDnet
from models import LEDNet
from models import unetformer
from models import EMALEDNet
from models import GAMLEDNET
from models import unetformer2
from models import mobilenetv2
from models import nestedunet1
from models import res_nestedunet
from models import efficientnet_nestedunet
from models import MobileNetV3_resunet
from models import EMA_RES_nested
from models import CGAFusion_resnestedunet
from models import shuffie_resnestedunet
from models import nestedunet_eca
from models import GCT_REESNESTEDUNET
from models import EMA_GCT_resnestedunet
from models import resnet50_nestedunet
from models import InceptionV4_nestedunet
from models import NestedUNet
def get_model(model_name):
    if model_name == 'UNet':
        model_ = UNet.UNet(in_channels=3, out_channels=nc)
    elif model_name == 'DeepLabV3Plus':
        model_ = deeplabv3_plus.DeepLab(num_classes=nc, backbone='mobilenet')
    elif model_name == 'VGG16':
        model_ = VGGNet.VGGNet16(num_classes=1)
        #model_ = vgg16(pretrained=False)  # 加载预训练的 VGG16 模型
        #num_features = model_.classifier[6].in_features
        #model_.classifier[6] = nn.Linear(num_features, 1)  # 修改最后一层全连接层输出为1维
    elif model_name == 'ResNet':
        model_ = ResNet.ResNet50(num_classes=2)  # 加载预训练的 ResNet-50 模型
        # 修改最后一层全连接层输出为指定的类别数
        num_features = model_.fc.in_features
        model_.fc = nn.Linear(num_features, nc)
    elif model_name == 'SegNet':
        model_ = SegNet.SegNet(num_classes= nc)
    elif model_name == 'unet++':
        model_ = nestedunet1.NestedUNet(in_channels=3,out_channels=nc)
    elif model_name == 'UNet++':
        model_ = NestedUNet.NestedUNet(in_channels=3,out_channels=nc)
    elif model_name == 'res_nestedunet':
        model_ = res_nestedunet.NestedUNet(in_channels=3,out_channels=nc)
    elif model_name == 'resnestedunet_eca':
        model_ = nestedunet_eca.NestedUNet(in_channels=3,out_channels=nc)
    elif model_name == 'efficient_nestedunet':
        model_ = efficientnet_nestedunet.NestedUNet(in_channels=3,out_channels=nc)
    elif model_name == 'EMA_GCT_resnestedunet':
        model_ = EMA_GCT_resnestedunet.NestedUNet(in_channels=3,out_channels=nc)
    elif model_name == 'CGAFusion_resnestedunet':
        model_ = CGAFusion_resnestedunet.NestedUNet(in_channels=3,out_channels=nc)
    elif model_name == 'resnet50_nestedunet':
        model_ = resnet50_nestedunet.NestedUNet(in_channels=3,out_channels=nc)
    elif model_name == 'GCT_resnestedunet':
        model_ = GCT_REESNESTEDUNET.NestedUNet(in_channels=3,out_channels=nc)
    elif model_name == 'inceptionv4_nestedunet':
        model_ = InceptionV4_nestedunet.NestedUNet(in_channels=3,out_channels=nc)
    elif model_name == 'EMA_RES_unet++':
        model_ = EMA_RES_nested.NestedUNet(in_channels=3,out_channels=nc)
    elif model_name == 'CBAMLEDNet':
        model_ = CBAMLEDnet.LEDNet(nclass=nc)
    elif model_name == 'CMLEDNet':
        model_ = CMLEDNet.LEDNet(nclass=nc)
    elif model_name == 'LEDNet':
        model_ = LEDNet.LEDNet(nclass = nc)
    elif model_name == 'unetformer':
        model_ = unetformer.UNetFormer(pretrained=False,num_classes=nc)
    elif model_name == 'unetformer2':
        model_ = unetformer2.UNetFormer(pretrained=False,num_classes=nc)
    elif model_name == 'GAMLEDNet':
        model_ = GAMLEDNET.LEDNet(nclass = nc)
    elif model_name == 'GAMMLEDNet':
        model_ = LEDNet2.LEDNet(nclass = nc)
    elif model_name == 'mobilenetv2':
        model_ = mobilenetv2.MobileNetV2(n_class=nc)
     #elif model_name == 'HATLEDNET':
        #model_ = HATLEDNET.LEDNet(nclass = nc)
    else:
        raise ValueError("Unsupported model name: {}".format(model_name))

    return model_