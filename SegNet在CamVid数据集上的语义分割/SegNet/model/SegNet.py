##################################################################################
#SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
#Paper-Link: https://arxiv.org/pdf/1511.00561.pdf
##################################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
#from tools.carafe import *
from tools.dysample import *


__all__ = ["SegNet"]

class SegNet(nn.Module):
    def __init__(self,classes= 19):
        super(SegNet, self).__init__()

        batchNorm_momentum = 0.1

        self.conv11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.downsample1 = Dy_DownSample(64)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        #self.downsample2 = CARAFE_downsample(c_in=128, c_mid=64, encoder_size=5, ratio=2, k_up=5)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        #self.downsample3 = CARAFE_downsample(c_in=256, c_mid=64, encoder_size=5, ratio=2, k_up=5)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        #self.downsample4 = CARAFE_downsample(c_in=512, c_mid=64, encoder_size=5, ratio=2, k_up=5)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        #self.downsample5 = CARAFE_downsample(c_in=512, c_mid=64, encoder_size=5, ratio=2, k_up=5)

        
        #self.upsample1 = CARAFE_upsample(c_in=512, c_mid=64, encoder_size=5, ratio=2, k_up=5)
        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        #self.upsample2 = CARAFE_upsample(c_in=512, c_mid=64, encoder_size=5, ratio=2, k_up=5)
        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

        #self.upsample3 = CARAFE_upsample(c_in=256, c_mid=64, encoder_size=5, ratio=2, k_up=5)
        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv31d = nn.Conv2d(256,  128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

        #self.upsample4 = CARAFE_upsample(c_in=128, c_mid=64, encoder_size=5, ratio=2, k_up=5)
        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

        #self.upsample5 = CARAFE_upsample(c_in=64, c_mid=64, encoder_size=5, ratio=2, k_up=5)
        self.upsample5 = Dy_UpSample(64)
        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv11d = nn.Conv2d(64, classes, kernel_size=3, padding=1)


    def forward(self, x):

        # Stage 1
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x1_size = x.size()
        #x, id1 = F.max_pool2d(x,kernel_size=2, stride=2,return_indices=True)
        x, off1 = self.downsample1(x)
        # x1p=self.downsample1(x12)
        #x=self.downsample1(x)

        # Stage 2
        # x21 = F.relu(self.bn21(self.conv21(x1p)))
        x = F.relu(self.bn21(self.conv21(x)))
        x = F.relu(self.bn22(self.conv22(x)))
        x2_size = x.size()
        # x2p, id2 = F.max_pool2d(x22,kernel_size=2, stride=2,return_indices=True)
        x, id2 = F.max_pool2d(x,kernel_size=2, stride=2,return_indices=True)
        #x=self.downsample2(x)

        # Stage 3
        # x31 = F.relu(self.bn31(self.conv31(x2p)))
        x = F.relu(self.bn31(self.conv31(x)))
        x = F.relu(self.bn32(self.conv32(x)))
        x = F.relu(self.bn33(self.conv33(x)))
        x3_size = x.size()
        # x3p, id3 = F.max_pool2d(x33,kernel_size=2, stride=2,return_indices=True)
        x, id3 = F.max_pool2d(x,kernel_size=2, stride=2,return_indices=True)
        #x=self.downsample3(x)

        # Stage 4
        # x41 = F.relu(self.bn41(self.conv41(x3p)))
        x = F.relu(self.bn41(self.conv41(x)))
        x = F.relu(self.bn42(self.conv42(x)))
        x = F.relu(self.bn43(self.conv43(x)))
        x4_size = x.size()
        # x4p, id4 = F.max_pool2d(x43,kernel_size=2, stride=2,return_indices=True)
        x, id4 = F.max_pool2d(x,kernel_size=2, stride=2,return_indices=True)
        #x=self.downsample4(x)

        # Stage 5
        # x51 = F.relu(self.bn51(self.conv51(x4p)))
        x = F.relu(self.bn51(self.conv51(x)))
        x = F.relu(self.bn52(self.conv52(x)))
        x = F.relu(self.bn53(self.conv53(x)))
        x5_size = x.size()
        # x5p, id5 = F.max_pool2d(x53,kernel_size=2, stride=2,return_indices=True)
        x, id5 = F.max_pool2d(x,kernel_size=2, stride=2,return_indices=True)
        #x=self.downsample5(x)


        # Stage 5d
        x = F.max_unpool2d(x, id5, kernel_size=2, stride=2, output_size=x5_size)
        #x = self.upsample1(x)
        # x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x5 = F.relu(self.bn53d(self.conv53d(x)))
        x5 = F.relu(self.bn52d(self.conv52d(x5)))
        x = F.relu(self.bn51d(self.conv51d(x)))

        # Stage 4d
        # x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2, output_size=x4_size)
        x = F.max_unpool2d(x, id4, kernel_size=2, stride=2, output_size=x4_size)
        #x = self.upsample2(x)
        #x = F.interpolate(x, size=(45, 60), mode='bilinear', align_corners=False) #填充为45*60
        # x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x = F.relu(self.bn43d(self.conv43d(x)))
        x = F.relu(self.bn42d(self.conv42d(x)))
        x = F.relu(self.bn41d(self.conv41d(x)))

        # Stage 3d
        # x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2, output_size=x3_size)
        x = F.max_unpool2d(x, id3, kernel_size=2, stride=2, output_size=x3_size)
        #x = self.upsample3(x)
        # x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x = F.relu(self.bn33d(self.conv33d(x)))
        x = F.relu(self.bn32d(self.conv32d(x)))
        x = F.relu(self.bn31d(self.conv31d(x)))

        # Stage 2d
        # x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2, output_size=x2_size)
        #x = self.upsample4(x)
        x = F.max_unpool2d(x, id2, kernel_size=2, stride=2, output_size=x2_size)
        # x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x = F.relu(self.bn22d(self.conv22d(x)))
        x = F.relu(self.bn21d(self.conv21d(x)))

        # Stage 1d
        #x = F.max_unpool2d(x, id1, kernel_size=2, stride=2, output_size=x1_size)
        x = self.upsample5(x,off1)
        #x = self.upsample5(x)
        x = F.relu(self.bn12d(self.conv12d(x)))
        x = self.conv11d(x)

        return x

    def load_from_segnet(self, model_path):
        s_dict = self.state_dict()# create a copy of the state dict
        th = torch.load(model_path).state_dict() # load the weigths
        # for name in th:
            # s_dict[corresp_name[name]] = th[name]
        self.load_state_dict(th)



"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegNet(classes=19).to(device)
    summary(model,(3,512,1024))