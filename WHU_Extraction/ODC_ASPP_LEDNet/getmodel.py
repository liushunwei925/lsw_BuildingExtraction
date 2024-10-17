from config import *

from models import lednet
def get_model(model_name):

    model_ = lednet.LEDNet(2)

    return model_