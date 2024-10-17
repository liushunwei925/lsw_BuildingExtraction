from config import *
from models import UNet
from models import NestedUNet,UNet
from models import SegNet
from models import lednet
from models import unetformer
from models import deeplabv3_plus
from models import CBAMLEDnet
from models import GCT_REESNESTEDUNET
from models import CGNet
from models import ASPPLEDNet
from models import OD_ASPP_LEDNet
from models import topformer
from models import segformer
from models import OD_SA_ASPP_LEDNet
def get_model(model_name):
    if model_name == "UNet":
        model_ = UNet.UNet(in_channels=3, out_channels=nc)
    elif model_name == "NestedUNet":
        model_ = NestedUNet.NestedUNet(in_channels=3, out_channels=nc)
    elif model_name == "GCT_resnestedunet":
        model_ = GCT_REESNESTEDUNET.NestedUNet(in_channels=3, out_channels=nc)
    elif model_name == "SegNet":
        model_ = SegNet.SegNet(num_classes=nc)
    #elif model_name == 'UNet':
        #model_ = UNet.UNet(in_channels=3,out_channels=nc)
    elif model_name == 'lednet':
        model_ = lednet.LEDNet(nclass=nc)
    elif model_name == 'OD_SA_ASPP_LEDNet':
        model_ = OD_SA_ASPP_LEDNet.LEDNet(nclass=nc)
    elif model_name == 'OD_ASPP_LEDNet':
        model_ = lednet.LEDNet(nclass=nc)
    elif model_name == 'ASPPLEDNet':
        model_ = ASPPLEDNet.LEDNet(nclass=nc)
    elif model_name == 'unetformer':
        model_ = unetformer.UNetFormer(pretrained=False,num_classes=nc)
    elif model_name == 'deeplabv3+':
        model_ = deeplabv3_plus.DeepLab(num_classes=nc, backbone='mobilenet')
    elif model_name == 'CBAMLEDNet':
        model_ = CBAMLEDnet.LEDNet(nclass=nc)
    elif model_name == "CGnet":
        model_ = CGNet.Context_Guided_Network(2)
    elif model_name == "topformer":
        model_ = topformer.TopFormer(nc)
    elif model_name =="segformer":
        model_ = segformer.SegFormer(num_classes=2)
    return model_