import torch.nn as nn
import torch
import torch.nn.functional as F
from hlindex import HolisticIndexBlock, DepthwiseO2OIndexBlock, DepthwiseM2OIndexBlock
from synergy_dysample import co_dysample
from synergy_carafe import Synergy_CARAFE
from norm.dysample import norm_dysample
from dysample import DySample
#from co import Dy_DownSample, Dy_UpSample
from dot.dysample import Co_dysample
from synergic_center.synergy_dysample import SynergicGenerator,Dysample


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = F.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c*block_size**2, h//block_size, w//block_size)


class RecoverNet(nn.Module):
    def __init__(self, downsampler,upsampler):
        super(RecoverNet, self).__init__()
        self.downsampler = downsampler
        self.upsampler =upsampler
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32*4 if self.downsampler=='s2d' else 32, 64, kernel_size=3, stride= 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64*4 if self.downsampler=='s2d' else 64, 128, kernel_size=3, stride= 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.midlayer1 = nn.Sequential(
            nn.Conv2d(128*4 if self.downsampler=='s2d' else 128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.midlayer2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.uplayer1 = nn.Sequential(
            nn.Conv2d(128//4 if self.upsampler=='d2s' else 128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.uplayer2 = nn.Sequential(
            nn.Conv2d(64//4 if self.upsampler=='d2s' else 64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )


        self.uplayer3 = nn.Conv2d(32//4 if self.upsampler=='d2s' else 32, 1, kernel_size=3, stride=1, padding=1)



        if self.downsampler=='conv':
            self.downsample1 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU()
            )
            self.downsample2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
            self.downsample3 = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        elif self.downsampler == 'maxpooling':
            self.downsample1 = nn.MaxPool2d((2, 2), stride=2, padding=0, return_indices=True)
            self.downsample2 = nn.MaxPool2d((2, 2), stride=2, padding=0, return_indices=True)
            self.downsample3 = nn.MaxPool2d((2, 2), stride=2, padding=0, return_indices=True)

        elif self.downsampler=='temp':
            self.downsample1 = Dy_DownSample(32)
            self.downsample2 = Dy_DownSample(64)
            self.downsample3 = Dy_DownSample(128)
        elif self.downsampler=='avgpool':
            self.downsample1 = nn.AvgPool2d((2, 2), stride=2, padding=0)
            self.downsample2 = nn.AvgPool2d((2, 2), stride=2, padding=0)
            self.downsample3 = nn.AvgPool2d((2, 2), stride=2, padding=0)




        if self.upsampler=='deconv':
            self.upsample1 = nn.Sequential(
                nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
            self.upsample2 = nn.Sequential(
                nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
            self.upsample3 = nn.Sequential(
                nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU()
            )
        elif self.upsampler=='maxunpooling':
            self.upsample1 = nn.MaxUnpool2d((2, 2), stride=2)
            self.upsample2 = nn.MaxUnpool2d((2, 2), stride=2)
            self.upsample3 = nn.MaxUnpool2d((2, 2), stride=2)

        elif self.upsampler=='temp':
            self.upsample1 = Dy_UpSample(128)
            self.upsample2 = Dy_UpSample(64)
            self.upsample3 = Dy_UpSample(32)

        elif self.upsampler=="index":
            self.index_block = DepthwiseM2OIndexBlock
            self.downsample1 = self.index_block(32, use_nonlinear=True, use_context=True,
                                             batch_norm=nn.BatchNorm2d)
            self.downsample2 = self.index_block(64, use_nonlinear=True, use_context=True,
                                            batch_norm=nn.BatchNorm2d)
            self.downsample3 = self.index_block(128, use_nonlinear=True, use_context=True,
                                            batch_norm=nn.BatchNorm2d)

        elif self.upsampler=='synergy_dysample':
            self.cosample1 = co_dysample(32)
            self.cosample2 = co_dysample(64)
            self.cosample3 = co_dysample(128)

        # elif self.upsampler=='carafe':
        #     self.upsample1 = CARAFE(128, 64)
        #     self.upsample2 = CARAFE(64, 32)
        #     self.upsample3 = CARAFE(32, 16)

        elif self.upsampler=='dysample':
            self.upsample1=DySample(128)
            self.upsample2=DySample(64)
            self.upsample3=DySample(32)

        elif self.upsampler=='co_dysample':
            self.cosample1 = Co_dysample(32)
            self.cosample2 = Co_dysample(64)
            self.cosample3 = Co_dysample(128)

        elif self.upsampler=='norm_dysample':
            self.cosample1 = norm_dysample(32)
            self.cosample2 = norm_dysample(64)
            self.cosample3 = norm_dysample(128)


        self.init_weights()


    def forward(self, input):
        x1 = self.layer1(input)

        if self.downsampler=='s2d':
            x1 = space_to_depth(x1, 2)
        elif self.downsampler=='synergy_dysample'or self.downsampler =='norm_dysample'or self.downsampler =='co_dysample':
            x1=self.cosample1.down_forward(x1)
        elif self.downsampler == 'maxpooling':
            x1,idx1= self.downsample1(x1)
        elif self.downsampler == 'conv':
            x1 = self.downsample1(x1)
        elif self.downsampler == 'temp':
            x1,off1 = self.downsample1(x1)
        elif self.downsampler == 'index':
            idx1_en, idx1_de = self.downsample1(x1)
            x1 = idx1_en * x1
            x1 = 4 * F.avg_pool2d(x1, (2, 2), stride=2)



        x2 = self.layer2(x1)
        if self.downsampler=='s2d':
            x2 = space_to_depth(x2, 2)
        elif self.downsampler=='synergy_dysample'or self.downsampler =='norm_dysample'or self.downsampler =='co_dysample':
            x2=self.cosample2.down_forward(x2)
        elif self.downsampler == 'maxpooling':
            x2, idx2 = self.downsample2(x2)
        elif self.downsampler == 'conv':
            x2 = self.downsample2(x2)
        elif self.downsampler == 'temp':
            x2,off2 = self.downsample2(x2)
        elif self.downsampler == 'index':
            idx2_en, idx2_de = self.downsample2(x2)
            x2 = idx2_en * x2
            x2 = 4 * F.avg_pool2d(x2, (2, 2), stride=2)


        x3 = self.layer3(x2)

        if self.downsampler=='s2d':
            x3 = space_to_depth(x3, 2)
        elif self.downsampler=='synergy_dysample'or self.downsampler =='norm_dysample'or self.downsampler =='co_dysample':
            x3=self.cosample3.down_forward(x3)
        elif self.downsampler == 'maxpooling':
            x3, idx3 = self.downsample3(x3)
        elif self.downsampler == 'conv':
            x3 = self.downsample3(x3)
        elif self.downsampler == 'temp':
            x3,off3 = self.downsample3(x3)
        elif self.downsampler == 'index':
            idx3_en, idx3_de = self.downsample3(x3)
            x3 = idx3_en * x3
            x3 = 4 * F.avg_pool2d(x3, (2, 2), stride=2)


        l = self.midlayer1(x3)
        l = self.midlayer2(l)

        if self.upsampler == 'd2s':
            l = F.pixel_shuffle(l, 2)
        elif self.upsampler == 'bilinear':
            l = F.interpolate(l, size=x2.size()[2:], mode='bilinear', align_corners=False)
        elif self.upsampler=='deconv'or self.upsampler=='carafe'or self.upsampler=='dysample':
            l = self.upsample1(l)
        elif self.upsampler=='temp':
            l = self.upsample1(l,off3)
        elif self.upsampler=='synergy_dysample'or self.upsampler=='norm_dysample'or self.upsampler=='co_dysample':
            l = self.cosample3.up_forward(l)
        elif self.upsampler == 'maxunpooling':
            l = self.upsample1(l, idx3)
        elif self.upsampler == 'nn':
            l = F.interpolate(l, size=(int(input.size()[2]/4), int(input.size()[3]/4)), mode='nearest')
        elif self.upsampler == 'index':
            l = idx3_de * F.interpolate(l, size=(int(input.size()[2]/4), int(input.size()[3]/4)), mode='nearest')


        l = self.uplayer1(l)
        if self.upsampler == 'd2s':
            l = F.pixel_shuffle(l, 2)
        elif self.upsampler == 'bilinear':
            l = F.interpolate(l, size=x1.size()[2:], mode='bilinear', align_corners=False)
        elif self.upsampler=='deconv'or self.upsampler=='carafe'or self.upsampler=='dysample':
            l = self.upsample2(l)
        elif self.upsampler=='temp':
            l = self.upsample2(l,off2)
        elif self.upsampler=='synergy_dysample'or self.upsampler=='norm_dysample'or self.upsampler=='co_dysample':
            l = self.cosample2.up_forward(l)
        elif self.upsampler == 'maxunpooling':
            l = self.upsample2(l, idx2)
        elif self.upsampler == 'nn':
            l = F.interpolate(l, size=(int(input.size()[2] / 2), int(input.size()[3] / 2)), mode='nearest')
        elif self.upsampler == 'index':
            l = idx2_de * F.interpolate(l, size=(int(input.size()[2]/2), int(input.size()[3]/2)), mode='nearest')



        l = self.uplayer2(l)
        if self.upsampler == 'd2s':
            l = F.pixel_shuffle(l, 2)
        elif self.upsampler == 'bilinear':
            l = F.interpolate(l, size=input.size()[2:], mode='bilinear', align_corners=False)
        elif self.upsampler=='deconv'or self.upsampler=='carafe'or self.upsampler=='dysample':
            l = self.upsample3(l)
        elif self.upsampler=='synergy_dysample'or self.upsampler=='norm_dysample'or self.upsampler=='co_dysample':
            l = self.cosample1.up_forward(l)
        elif self.upsampler=='temp':
            l = self.upsample3(l,off1)
        elif self.upsampler == 'maxunpooling':
            l = self.upsample3(l, idx1)
        elif self.upsampler == 'nn':
            l = F.interpolate(l, size=input.size()[2:], mode='nearest')
        elif self.upsampler == 'index':
            l = idx1_de * F.interpolate(l, size=input.size()[2:], mode='nearest')


        out = self.uplayer3(l)

        return out

    def init_weights(self, init_type='normal', init_gain=0.02):
        """
        Parameters:
            net (network)   -- network to be initialized
            init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
        """

        for m in self.modules():
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif classname.find(
                    'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
                torch.nn.init.constant_(m.bias.data, 0.0)

