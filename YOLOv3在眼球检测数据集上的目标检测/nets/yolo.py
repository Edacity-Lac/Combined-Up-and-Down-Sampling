from collections import OrderedDict

import torch
import torch.nn as nn

from nets.darknet import darknet53

import torch.nn.functional as F

from co_dysample import Dy_UpSample, Dy_DownSample
import einops
from hlindex import DepthwiseO2OIndexBlock, DepthwiseM2OIndexBlock


pool_indices=[0,0]
offset=[0,0]
index_decoder=[0,0]

def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = F.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c*block_size**2, h//block_size, w//block_size)

class CARAFE_downsample(nn.Module):
    def __init__(self, c_in, c_mid=64, encoder_size=5, ratio=2, k_up=5):
        super().__init__()
        padding = 2  # 只允许 encoder_size=5, ratio=2
        self.ratio = ratio
        self.k_up = k_up
        self.unfold_padding = 2  # k_up=5, ratio=2
        self.comp = nn.Conv2d(c_in, c_mid, kernel_size=1, padding=0)
        self.content_encoder = nn.Conv2d(c_mid, k_up ** 2, kernel_size=encoder_size,
                                         stride=ratio, padding=padding)

    def forward(self, x):
        _, _, _, W = x.size()
        y = self.comp(x)
        y = self.content_encoder(y)
        y = F.softmax(y, dim=1)

        z = F.unfold(x, kernel_size=self.k_up, stride=self.ratio, padding=self.unfold_padding)
        z = einops.rearrange(z, 'b (c k_up2) (h w) -> b k_up2 c h w',
                             k_up2=self.k_up ** 2, w=W // self.ratio)
        x = torch.einsum('bkchw,bkhw->bchw', [z, y])
        return x


class CARAFE_upsample(nn.Module):
    def __init__(self, c_in, c_mid=64, encoder_size=5, ratio=2, k_up=5):
        super().__init__()
        padding = (encoder_size - 1) // 2 if encoder_size % 2 == 0 else encoder_size // 2
        self.ratio = ratio
        self.k_up = k_up
        self.unfold_padding = (k_up - 1) // 2 if k_up % 2 == 0 else k_up // 2
        self.comp = nn.Conv2d(c_in, c_mid, kernel_size=1, padding=0)
        self.content_encoder = nn.Conv2d(c_mid, (ratio * k_up) ** 2, kernel_size=encoder_size,
                                         padding=padding)

    def forward(self, x):
        _, _, _, W = x.size()
        y = self.comp(x)
        y = self.content_encoder(y)
        y = F.pixel_shuffle(y, upscale_factor=self.ratio)
        y = F.softmax(y, dim=1)

        z = F.unfold(x, kernel_size=self.k_up, padding=self.unfold_padding)
        z = einops.rearrange(z, 'b (c k_up2) (h w) -> b k_up2 c h w',
                             k_up2=self.k_up ** 2, w=W)
        z = einops.repeat(z, 'b k c h w -> ratio_2 b k c h w', ratio_2=self.ratio ** 2)
        z = einops.rearrange(z, '(r1 r2) b k_up2 c h w -> b k_up2 c (h r1) (w r2)',
                             r1=self.ratio)
        x = torch.einsum('bkchw,bkhw->bchw', [z, y])
        return x

class UpsampleCNN(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_scale_factor=2):
        super(UpsampleCNN, self).__init__()
        
        # 假设我们只使用一个卷积层作为示例
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # upsample_scale_factor定义了上采样的缩放因子
        self.upsample_scale_factor = upsample_scale_factor
    
    def forward(self, x):
        # 假设x是通过某个网络层得到的特征图
        x = self.conv(x)
        
        # 使用双线性插值进行上采样
        x = F.interpolate(x, scale_factor=self.upsample_scale_factor, mode='bilinear', align_corners=True)
        
        # 返回上采样后的特征图
        return x



def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))


def conv2d2(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    if kernel_size==3:
        return nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(filter_in, filter_out, kernel_size=1, stride=1, padding=pad, bias=False),
            nn.BatchNorm2d(filter_out),
            nn.LeakyReLU(0.1),
        )
    else:
        return nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
            ("bn", nn.BatchNorm2d(filter_out)),
            ("relu", nn.LeakyReLU(0.1)),
        ]))

#------------------------------------------------------------------------#
#   make_last_layers里面一共有七个卷积，前五个用于提取特征。
#   后两个用于获得yolo网络的预测结果
#------------------------------------------------------------------------#
def make_last_layers(filters_list, in_filters, out_filter):
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
    )
    return m

# def make_last_layers(filters_list, in_filters, out_filter):
#     m = nn.Sequential(
#         conv2d(in_filters, filters_list[0], 1),
#         nn.MaxPool2d(kernel_size=2, stride=2),
#         conv2d(filters_list[0], filters_list[1], 1),
#
#         conv2d(filters_list[1], filters_list[0], 1),
#         nn.MaxPool2d(kernel_size=2, stride=2),
#         conv2d(filters_list[0], filters_list[1], 1),
#
#         # conv2d(filters_list[1], filters_list[0], 1),
#         # nn.MaxPool2d(kernel_size=2, stride=2),
#         conv2d(filters_list[0], filters_list[1], 1),
#
#         nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
#     )
#     return m




class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, pretrained = False):
        super(YoloBody, self).__init__()
        #---------------------------------------------------#   
        #   生成darknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        #---------------------------------------------------#
        self.backbone = darknet53()
        # if pretrained:
        #     self.backbone.load_state_dict(torch.load("model_data/darknet53_backbone_weights.pth"))

        #---------------------------------------------------#
        #   out_filters : [64, 128, 256, 512, 1024]
        #---------------------------------------------------#
        out_filters = self.backbone.layers_out_filters







        # 标准
        # self.last_layer0            = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))
        #
        # self.last_layer1_conv       = conv2d(512, 256, 1)
        # self.last_layer1_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        # self.last_layer1            = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))
        #
        # self.last_layer2_conv       = conv2d(256, 128, 1)
        # self.last_layer2_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        # self.last_layer2            = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))
        #


        # 实验一 卷积和反卷积
        # self.last_layer0            = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))
        #
        # self.last_layer1_conv       = conv2d(512, 256, 1)
        # # self.last_layer1_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        #
        # # 这里要设置输入通道和输出通道的数目都是256，通过反卷积实现2倍上采样的效果
        # self.last_layer1_upsample   = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1,output_padding=1)
        #
        # self.last_layer1            = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))
        #
        # self.last_layer2_conv       = conv2d(256, 128, 1)
        # # self.last_layer2_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        #
        # # 这里要设置输入通道和输出通道的数目都是128，通过反卷积实现2倍上采样的效果
        # self.last_layer2_upsample   = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1,output_padding=1)
        #
        # self.last_layer2            = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))



        # 实验二，卷积和非线性插值
        # self.last_layer0            = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))
        #
        # self.last_layer1_conv       = conv2d(512, 256, 1)
        # # self.last_layer1_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        # self.last_layer1_upsample   = UpsampleCNN(in_channels=256, out_channels=256, upsample_scale_factor=2)
        #
        #
        #
        # self.last_layer1            = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))
        #
        # self.last_layer2_conv       = conv2d(256, 128, 1)
        # # self.last_layer2_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        # self.last_layer2_upsample   = UpsampleCNN(in_channels=128, out_channels=128, upsample_scale_factor=2)
        #
        #
        # self.last_layer2            = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))
        #

        



        # # 实验三，最大池化和标准上采采样
        # self.last_layer0            = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))
        #
        # self.last_layer1_conv       = conv2d(512, 256, 1)
        # self.last_layer1_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        # self.last_layer1            = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))
        #
        # self.last_layer2_conv       = conv2d(256, 128, 1)
        # self.last_layer2_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        # self.last_layer2            = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))
        #
        # # 实验四，carafe++
        # self.last_layer0 = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))
        # self.last_layer1_conv = conv2d(512, 256, 1)
        # self.last_layer1_upsample =  CARAFE_upsample(256, 64, 5, ratio=2)
        # self.last_layer1 = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))
        # self.last_layer2_conv = conv2d(256, 128, 1)
        # self.last_layer2_upsample =CARAFE_upsample(128, 64, 5, ratio=2)
        # self.last_layer2 = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))

        # # 实验五，spacetodepth
        # self.last_layer0 = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))
        #
        # self.last_layer1_conv = conv2d(512, 256, 1)
        # self.last_layer1_upsample = conv2d(256*4,256,1)#space to depth的上采样在forward里，此处用于通道数复原
        # self.last_layer1 = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))
        #
        # self.last_layer2_conv = conv2d(256, 128, 1)
        # self.last_layer2_upsample = conv2d(128*4,128,1)#space to depth的上采样在forward里，此处用于通道数复原
        # self.last_layer2 = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))


        # # 实验六 CO-Dysample
        # self.last_layer0 = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))
        # self.last_layer1_conv = conv2d(512, 256, 1)
        # self.last_layer1_upsample =   Dy_UpSample(256,'lp', 2, 4, True)
        # self.last_layer1 = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))
        # self.last_layer2_conv = conv2d(256, 128, 1)
        # self.last_layer2_upsample  = Dy_UpSample(128, 'lp', 2, 4, True)
        # self.last_layer2 = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))

        # # 实验六 CO-Dysample
        # self.last_layer0 = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))
        # self.last_layer1_conv = conv2d(512, 256, 1)
        # self.last_layer1_upsample = Dy_UpSample(256, 'lp', 2, 4, True)
        # self.last_layer1 = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))
        # self.last_layer2_conv = conv2d(256, 128, 1)
        # self.last_layer2_upsample = Dy_UpSample(128, 'lp', 2, 4, True)
        # self.last_layer2 = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))
        # 实验七  index sampling
        self.last_layer0            = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))

        self.last_layer1_conv       = conv2d(512, 256, 1)
        # self.last_layer1_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1_upsample   = UpsampleCNN(in_channels=256, out_channels=256, upsample_scale_factor=2)



        self.last_layer1            = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))

        self.last_layer2_conv       = conv2d(256, 128, 1)
        # self.last_layer2_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2_upsample   = UpsampleCNN(in_channels=128, out_channels=128, upsample_scale_factor=2)


        self.last_layer2            = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))


    def forward(self, x):
        #---------------------------------------------------#   
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256；26,26,512；13,13,1024
        #---------------------------------------------------#
        x2, x1, x0 = self.backbone(x)

        #---------------------------------------------------#
        #   第一个特征层
        #   out0 = (batch_size,255,13,13)
        #---------------------------------------------------#
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        out0_branch = self.last_layer0[:5](x0)
        out0        = self.last_layer0[5:](out0_branch)

        # 13,13,512 -> 13,13,256 -> 26,26,256xiugaisuanzide1t
        x1_in = self.last_layer1_conv(out0_branch)
        # 如果是spacetodepth   执行下一行
        # x1_in = space_to_depth(x1_in,2)
        x1_in = self.last_layer1_upsample(x1_in)
        # 如果是codysample：
        # x1_i = self.last_layer1_upsample(x1_in, offset[1])
        # 如果是index
        # x1_in = index_decoder[1]*self.last_layer1_upsample(x1_in)

        # 26,26,256 + 26,26,512 -> 26,26,768
        
        print("x1_in.shape:",x1_in.shape)
        print("x1.shape:",x1.shape)
        
        x1_in = torch.cat([x1_in, x1], 1)
        #---------------------------------------------------#
        #   第二个特征层
        #   out1 = (batch_size,255,26,26)
        #---------------------------------------------------#
        # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        out1_branch = self.last_layer1[:5](x1_in)
        out1        = self.last_layer1[5:](out1_branch)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        x2_in = self.last_layer2_conv(out1_branch)
        # 如果是spacetodepth   执行下一行
        # x2_in = space_to_depth(x2_in,2)
        x2_in = self.last_layer2_upsample(x2_in)
        # 如果是codysample：
        # x2_i = self.last_layer2_upsample(x2_in, offset[0])
        # 如果是index
        # x2_in = index_decoder[0] * self.last_layer2_upsample(x2_in)
        # 52,52,128 + 52,52,256 -> 52,52,384
        x2_in = torch.cat([x2_in, x2], 1)
        #---------------------------------------------------#
        #   第一个特征层
        #   out3 = (batch_size,255,52,52)
        #---------------------------------------------------#
        # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        out2 = self.last_layer2(x2_in)
        return out0, out1, out2