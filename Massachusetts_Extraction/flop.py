from config import *
from models import lednet
from torchstat import stat
model = lednet.LEDNet(nclass= nc)

stat(model, (3,256, 256))