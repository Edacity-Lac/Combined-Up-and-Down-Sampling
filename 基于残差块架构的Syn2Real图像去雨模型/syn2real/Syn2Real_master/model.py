import pdb
import math
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable
from hlindex import HolisticIndexBlock, DepthwiseO2OIndexBlock, DepthwiseM2OIndexBlock

#indices = torch.tensor([1, 2, 3])

#def modify_global_var(new_indices):
 #   global indices
 #   indices=new_indices

class CARAFE(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, kernel_size=3, group_channels=1):
        super(CARAFE, self).__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels * scale_factor ** 2, kernel_size=kernel_size, padding=kernel_size // 2)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        return x


def space2depth(input_tensor, block_size):
    batch_size, channels, height, width = input_tensor.size()
    assert height % block_size == 0 and width % block_size == 0, \
        "Input tensor's height and width must be divisible by block_size."

    unfolded_tensor = F.unfold(input_tensor, block_size, stride=block_size)
    unfolded_channels = channels * (block_size ** 2)
    unfolded_height = height // block_size
    unfolded_width = width // block_size

    output_tensor = unfolded_tensor.view(batch_size, unfolded_channels, unfolded_height, unfolded_width)

    return output_tensor



def depth2space(input_tensor, block_size):
    batch_size, channels, height, width = input_tensor.size()
    assert channels % (block_size ** 2) == 0, \
        "Number of input channels must be divisible by block_size squared."

    unfolded_channels = channels // (block_size ** 2)
    unfolded_height = height * block_size
    unfolded_width = width * block_size

    # Reshape the input tensor
    reshaped_tensor = input_tensor.view(batch_size, unfolded_channels, block_size, block_size, height, width)

    # Permute and reshape to get the desired output tensor
    output_tensor = reshaped_tensor.permute(0, 1, 4, 2, 5, 3).contiguous()
    output_tensor = output_tensor.view(batch_size, unfolded_channels, unfolded_height, unfolded_width)

    return output_tensor


def conv_block(in_dim,out_dim):
  return nn.Sequential(nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(in_dim,out_dim,kernel_size=1,stride=1,padding=0),
                       nn.AvgPool2d(kernel_size=2,stride=2))
def deconv_block(in_dim,out_dim):
  return nn.Sequential(nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.UpsamplingNearest2d(scale_factor=2))


#max pooling-max unpooling
#class MaxPool2dWithIndices(nn.Module):
#    def __init__(self, kernel_size=3, stride=1, padding=0):
 #       super(MaxPool2dWithIndices, self).__init__()
  #      self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, return_indices=True)

 #   def forward(self, x):
 #       output, indices = self.pool(x)
 #       return output, indices


class SEBlock(nn.Module):
    def __init__(self, input_dim, reduction):
        super().__init__()
        mid = int(input_dim / reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class Bottle2neckX(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
            scale: number of scale.
            type: 'normal': normal set. 'stage': frist block of a new stage.
        """
        super(Bottle2neckX, self).__init__()

        D = int(math.floor(planes * (baseWidth/64.0)))
        C = cardinality

        self.conv1 = nn.Conv2d(inplanes, D*C*scale, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(D*C*scale)
        self.SE = SEBlock(inplanes,C)

        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)  #downsample1/2
            #self.pool = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1, bias=False)  #conv-deconv
            #self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True) #max pooling-max unpooling
            #self.space2depth_stride = stride  #  space2depth
            #self.pool = None
            #self.pool = CARAFE(inplanes, inplanes, scale_factor=stride)  #CARAFE

        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(D*C, D*C, kernel_size=3, stride = stride, padding=1, groups=C, bias=False))
          bns.append(nn.BatchNorm2d(D*C))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(D*C*scale, inplanes  , kernel_size=1, stride=1, padding=0, bias=False)        
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.width  = D*C
        self.stype = stype
        self.scale = scale

    def forward(self, x):
        residual = x

        
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          
          sp = self.relu(self.bns[i](sp))
          sp = self.convs[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)
          #pooled_sp, new_indices = self.pool(spx[self.nums])   #max pooling-max unpooling
          #out = torch.cat((out, pooled_sp), 1)
          #modify_global_var(new_indices)
          #------------------------------------space2depth
          #out_space2depth = space2depth(spx[self.nums], self.space2depth_stride)
          #out = torch.cat((out, out_space2depth), 1)
        
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        #out = self.SE(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        #pdb.set_trace()
        out += residual


        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inter_planes)
        self.conv3 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)



class BottleneckBlock1(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock1, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=5, stride=1,
                               padding=2, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)



class BottleneckBlock2(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock2, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=7, stride=1,
                               padding=3, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


#TransitionBlock为上采样操作部分，在整个网络框架的第7，8，9块引用，进行上采样
class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
        self.carafe = CARAFE(out_planes, out_planes, scale_factor=2)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)  #max pooling-max unpooling


    def forward(self, x, ind):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        #return F.upsample_nearest(out, scale_factor=2)   #upsample
        #  conv-deconv
        #out = F.interpolate(out, scale_factor=2, mode='nearest')
        #return out
        # conv-bilinear
        #out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        #return out
        # max pooling-max unpooling
        #print("out shape:", out.shape)
        #out = self.unpool(out, indices)
        #return out
        #space2depth-depth2space
        #out = depth2space(out, 1)
        #return out
        #carafe
        #out = self.carafe(out)
        #return out
        #print(out.size())
        #print(x.size())
        #print(ind.size())
        out = ind*F.interpolate(out, scale_factor=2, mode='nearest')
        #out = ind * F.interpolate(out, size=(int(out.size()[2] / 4), int(out.size()[3] / 4)), mode='nearest')
        return out


#TransitionBlock1是下采样部分，在整个网络框架的2，3，4块引用，进行下采样
class TransitionBlock1(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock1, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        #self.downsample = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=2, padding=1) #conv-deconv
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)   ##max pooling-max unpooling
        self.carafe = CARAFE(out_planes, out_planes, scale_factor=2)  #carafe
        #--------------------------------
        self.index_block = DepthwiseM2OIndexBlock
        self.downsample1 = self.index_block(out_planes, use_nonlinear=True, use_context=True,
                                            batch_norm=nn.BatchNorm2d)  # index
        self.droprate = dropRate
        self.space2depth_stride = 1
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        #return F.avg_pool2d(out, 2)    #dowmsample
        #out = self.downsample(out)   #conv-deconv
        #return out
        #out, indices = self.pool(out)  #-----------
        #return  out,indices  #max pooling-max unpooling
        #out = space2depth(out, self.space2depth_stride)   #space2depth-depth2space
        #return out
        #out = self.carafe(out)
        #return out
        idx1_en, idx1_de = self.downsample1(out)
        out = idx1_en * out
        out = 4 * F.avg_pool2d(out, (2, 2), stride=2)
        return out,idx1_de

class TransitionBlock3(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock3, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)

        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out






class vgg19ca(nn.Module):
    def __init__(self):
        super(vgg19ca, self).__init__()




        ############# 256-256  ##############
        haze_class = models.vgg19_bn(pretrained=True)
        self.feature = nn.Sequential(haze_class.features[0])

        for i in range(1,3):
            self.feature.add_module(str(i),haze_class.features[i])

        self.conv16=nn.Conv2d(64, 24, kernel_size=3,stride=1,padding=1)  # 1mm
        self.dense_classifier=nn.Linear(127896, 512)
        self.dense_classifier1=nn.Linear(512, 4)


    def forward(self, x):

        feature=self.feature(x)
        # feature = Variable(feature.data, requires_grad=True)

        feature=self.conv16(feature)
        # print feature.size()

        # feature=Variable(feature.data,requires_grad=True)



        out = F.relu(feature, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7).view(out.size(0), -1)
        # print out.size()

        # out=Variable(out.data,requires_grad=True)
        out = F.relu(self.dense_classifier(out))
        out = (self.dense_classifier1(out))


        return out


class scale_residue_est(nn.Module):
    def __init__(self):
        super(scale_residue_est, self).__init__()

        self.conv1 = BottleneckBlock(64, 32)
        self.trans_block1 = TransitionBlock3(96, 32)
        self.conv2 = BottleneckBlock(32, 32)
        self.trans_block2 = TransitionBlock3(64, 32)
        self.conv3 = BottleneckBlock(32, 32)
        self.trans_block3 = TransitionBlock3(64, 32)
        self.conv_refin = nn.Conv2d(32, 16, 3, 1, 1)
        self.tanh = nn.Tanh()
        self.refine3 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1=self.conv1(x)
        x1 = self.trans_block1(x1)
        x2=self.conv2(x1)
        x2 = self.trans_block2(x2)
        x3=self.conv3(x2)
        x3 = self.trans_block3(x3)
        x4 = self.relu((self.conv_refin(x3)))
        residual = self.tanh(self.refine3(x4))

        return residual

class scale_residue_conf(nn.Module):
    def __init__(self):
        super(scale_residue_conf, self).__init__()

        self.conv1 = nn.Conv2d(35,16,3,1,1)#BottleneckBlock(35, 16)
        #self.trans_block1 = TransitionBlock3(51, 8)
        self.conv2 = BottleneckBlock(16, 16)
        self.trans_block2 = TransitionBlock3(32, 16)
        self.conv3 = BottleneckBlock(16, 16)
        self.trans_block3 = TransitionBlock3(32, 16)
        self.conv_refin = nn.Conv2d(16, 16, 3, 1, 1)
        self.sig = torch.nn.Sigmoid()
        self.refine3 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1=self.conv1(x)
        #x1 = self.trans_block1(x1)
        x2=self.conv2(x1)
        x2 = self.trans_block2(x2)
        x3=self.conv3(x2)
        x3 = self.trans_block3(x3)
        residual = self.sig(self.refine3(x3))

        return residual




#这是网络的主函数，分为9个结构相似的块。每个块包含一个残差块、上下采样算子及相应的维度转换函数等。
class DeRain_v2(nn.Module):
    def __init__(self):
        super(DeRain_v2, self).__init__()
        self.baseWidth = 12#4#16
        self.cardinality = 8#8#16
        self.scale = 6#4#5
        self.stride = 1
        ############# Block1-scale 1.0  ##############
        self.conv_input=nn.Conv2d(3,16,3,1,1)
        self.dense_block1 = Bottle2neckX(16,16, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')

        ############# Block2-scale 0.50  ##############
        self.trans_block2=TransitionBlock1(32,32)
        self.dense_block2 = Bottle2neckX(32,32, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        self.trans_block2_o=TransitionBlock3(64,32)

        ############# Block3-scale 0.250  ##############
        self.trans_block3=TransitionBlock1(32,32)
        self.dense_block3=Bottle2neckX(32,32, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        self.trans_block3_o=TransitionBlock3(64,64)

        ############# Block4-scale 0.25  ##############
        self.trans_block4=TransitionBlock1(64,128)
        self.dense_block4=Bottle2neckX(128,128, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        self.trans_block4_o=TransitionBlock3(256,128)

        ############# Block5-scale 0.25  ##############
        self.dense_block5=Bottle2neckX(128,128, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        self.trans_block5_o=TransitionBlock3(256,128)

        ############# Block6-scale 0.25  ##############
        self.dense_block6=Bottle2neckX(128,128, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        self.trans_block6_o=TransitionBlock3(256,128)

        ############# Block7-scale 0.25  ############## 7--3 skip connection
        #self.trans_block7=TransitionBlock(32,64)
        self.trans_block7 = TransitionBlock(32, 64)
        self.dense_block7=Bottle2neckX(128,128, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        self.trans_block7_o=TransitionBlock3(256,32)

        ############# Block8-scale 0.5  ############## 8--2 skip connection
        self.trans_block8=TransitionBlock(32,32)
        self.dense_block8=Bottle2neckX(64,64, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        self.trans_block8_o=TransitionBlock3(128,32)

        ############# Block9-scale 1.0  ############## 9--1 skip connection
        self.trans_block9=TransitionBlock(32,32)
        self.dense_block9=Bottle2neckX(80,80, self.baseWidth, self.cardinality, self.stride, downsample=None, scale=self.scale, stype='normal')
        self.trans_block9_o=TransitionBlock3(160,16)


        self.conv_refin=nn.Conv2d(16,16,3,1,1)
        self.tanh=nn.Tanh()
        self.sig=nn.Sigmoid()


        self.refine3= nn.Conv2d(16, 3, kernel_size=3,stride=1,padding=1)
        self.zout= nn.Conv2d(128, 32, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest  #upsample-------------------------------
        #self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.refineclean1= nn.Conv2d(3, 8, kernel_size=7,stride=1,padding=3)
        self.refineclean2= nn.Conv2d(8, 3, kernel_size=3,stride=1,padding=1)


        self.adjust3 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)



    def forward(self, xin):
        x= self.conv_input(xin)
        #Size - 1.0
        x1=(self.dense_block1(x))

        #Size - 0.5        
        x2_i,ind1=self.trans_block2(x1)
        #print("Indices1 shape:", ind1.shape)
        x2_i=self.dense_block2(x2_i)
        x2=self.trans_block2_o(x2_i)

        #Size - 0.25
        x3_i,ind2=self.trans_block3(x2)
        #print("Indices2 shape:", ind2.shape)
        x3_i=self.dense_block3(x3_i)
        x3=self.trans_block3_o(x3_i)

        #Size - 0.125
        x4_i,ind3=self.trans_block4(x3)
        #print("x3 shape:", x3.shape)
        #print("Indices3 shape:", ind3.shape)
        x4_i=self.dense_block4(x4_i)
        x4=self.trans_block4_o(x4_i)

        x5_i=self.dense_block5(x4)
        x5=self.trans_block5_o(x5_i)

        x6_i=self.dense_block6(x5)
        x6=self.trans_block6_o(x6_i)
        z = self.zout(self.relu(x6))


        ind3=ind3.float()
        ind3 = self.adjust3(ind3)
        #z = F.interpolate(z, size=(168, 248), mode='nearest')
        #z = torch.cat([z] * 2, dim=1)
        #print("x7 shape:",z.shape)
        ind3 = ind3.to(torch.int64)
        x7_i=self.trans_block7(z,ind3)
        # print(x4.size())
        # print(x7_i.size())
        x7_i=self.dense_block7(torch.cat([x7_i, x3], 1))
        x7=self.trans_block7_o(x7_i)

        x8_i=self.trans_block8(x7,ind2)
        x8_i=self.dense_block8(torch.cat([x8_i, x2], 1))
        x8=self.trans_block8_o(x8_i)

        x9_i=self.trans_block9(x8,ind1)
        x9_i=self.dense_block9(torch.cat([x9_i, x1,x], 1))
        x9=self.trans_block9_o(x9_i)

        # x10_i=self.trans_block10(x9)
        # x10_i=self.dense_block10(torch.cat([x10_i, x1,x], 1))
        # x10=self.trans_block10_o(x10_i)

        x11=x-self.relu((self.conv_refin(x9)))
        residual=self.tanh(self.refine3(x11))
        clean = residual
        clean=self.relu(self.refineclean1(clean))
        clean=self.sig(self.refineclean2(clean))

        return clean,z

def gradient(y):
    gradient_h=y[:, :, :, :-1] - y[:, :, :, 1:]
    gradient_v=y[:, :, :-1, :] - y[:, :, 1:, :]

    return gradient_h, gradient_v

def TV(y):
    gradient_h=torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
    gradient_v=torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])

    return gradient_h, gradient_v

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=64):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(6, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()