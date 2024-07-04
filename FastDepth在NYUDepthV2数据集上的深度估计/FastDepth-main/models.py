import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import math,time
import utils
from co_dysample import Dy_UpSample, Dy_DownSample
import einops
from hlindex import DepthwiseO2OIndexBlock, DepthwiseM2OIndexBlock


pool_indices=[0,0,0,0,0]
offset=[0,0,0,0,0]
index_decoder=[0,0,0,0,0]

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
def load_network_structure():
    args = utils.parse_args()
    return args.encoder_decoder_choice

my_choice = load_network_structure()


def DeconvBlock(in_channels,out_channels,kernel_size):
  return nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size,
                stride=2,padding = (kernel_size - 1) // 2,output_padding = kernel_size%2,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
def ConvBlock(in_channels,out_channels,kernel_size,stride,padding):
  return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

def ConvReLU6Block(in_channels,out_channels,kernel_size,stride,padding):
  return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )



def ConvReLU6BlockCarafe(in_channels,out_channels,kernel_size,stride,padding):
  return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,1,stride=1,padding=0,bias=False),
            CARAFE_downsample(out_channels, 64, 5, ratio=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

class ConvReLU6Blocks2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvReLU6Blocks2d, self).__init__()

        self.conv = nn.Conv2d(4*in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu6 = nn.ReLU6(inplace=True)
    def forward(self, x):
        # Apply MaxPool and get indices
        x = space_to_depth(x, 2)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu6(x)
        # Apply convolution, batch normalization, and ReLU6
        # Return the output and the indices
        return x

class ConvReLU6Blockindexed(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvReLU6Blockindexed, self).__init__()
        self.mydownsample = DepthwiseM2OIndexBlock(out_channels, use_nonlinear=True, use_context=True, batch_norm=nn.BatchNorm2d)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu6 = nn.ReLU6(inplace=True)
    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu6(x)
        idx1_en, idx1_de = self.mydownsample(x)
        x = idx1_en * x
        x = 4 * F.avg_pool2d(x, (2, 2), stride=2)

        return x,  idx1_de

class ConvReLU6BlockDysample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvReLU6BlockDysample, self).__init__()
        self.dy =Dy_DownSample(out_channels, 'lp', 2, 4, True)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu6 = nn.ReLU6(inplace=True)
    def forward(self, x):
        # Apply MaxPool and get indices
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu6(x)
        x, my_off = self.dy(x)
        # Apply convolution, batch normalization, and ReLU6
        # Return the output and the indices
        return x, my_off


class ConvReLU6BlockPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvReLU6BlockPool, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2, padding=0, return_indices=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu6 = nn.ReLU6(inplace=True)
    def forward(self, x):
        # Apply MaxPool and get indices

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu6(x)
        x, indices = self.pool(x)
        # Apply convolution, batch normalization, and ReLU6
        # Return the output and the indices
        return x, indices

def DWConvBlock(in_channels,out_channels,kernel_size,stride,padding = None):
  if padding == None:
    padding = (kernel_size - 1) // 2
    # padding = padding+1
  return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,bias=False,groups=in_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


def weights_init(m):
    # Initialize kernel weights with Gaussian distributions
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
        

class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            # print(x.size())
            # print(self.conv(x).size())
            # time.sleep(5)  # 休眠1秒

            return x + self.conv(x)
        else:
            return self.conv(x)

class InvertedResidualBlockCarafe(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidualBlockCarafe, self).__init__()
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dww
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        elif stride == 2:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                # nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                CARAFE_downsample(hidden_dim, 64, 5, ratio=2),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                # CARAFE_downsample(hidden_dim, 64, 5, ratio=2),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )


    def forward(self, x):
        if self.identity:
            # print(x.size())
            # print(self.conv(x).size())
            return x + self.conv(x)
        else:
            return self.conv(x)

class InvertedResidualBlockDysample(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidualBlockDysample, self).__init__()
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        self.return_offset =  expand_ratio != 1 and stride == 2

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        elif stride == 2:
            self.conv1 = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True) )

            self.dysample = Dy_DownSample(hidden_dim, 'lp', 2, 4, True)
            self.conv2 = nn.Sequential(

                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                # CARAFE_downsample(hidden_dim, 64, 5, ratio=2),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )


    def forward(self, x):
        if self.identity:
            # print(x.size())
            # print(self.conv(x).size())
            return x + self.conv(x)
        elif self.return_offset:
            x=self.conv1(x)
            x, a_offset=self.dysample(x)
            x=self.conv2(x)
            return x, a_offset
        else:
            return self.conv(x)


class InvertedResidualBlockS2d(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidualBlockS2d, self).__init__()
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        self.return_offset =  expand_ratio != 1 and stride == 2

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        elif stride == 2:
            self.conv1 = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
                # dw
                 )

            self.conv2 = nn.Sequential(
                nn.Conv2d(4*hidden_dim, hidden_dim, 1, 1, 0, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                # CARAFE_downsample(hidden_dim, 64, 5, ratio=2),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )


    def forward(self, x):
        if self.identity:
            # print(x.size())
            # print(self.conv(x).size())
            return x + self.conv(x)
        elif self.return_offset:
            x=self.conv1(x)
            x = space_to_depth(x,2)
            x=self.conv2(x)
            return x
        else:
            return self.conv(x)

class InvertedResidualBlockPool(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidualBlockPool, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        self.return_indices = expand_ratio != 1 and stride == 2

        if expand_ratio == 1:  # 3层以后都是6
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.conv1 = nn.Sequential(
            nn.Conv2d(oup, inp, 1, 1, 0,  bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0,  bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )
    def forward(self, x):
        if self.identity:  # 条件是步长为1，并且输入=输出
            return x + self.conv(x)
        elif self.return_indices:#条件是expand raio不是1，并且stride=2
            x = self.conv(x)
            x = self.conv1(x)
            x, indices = self.pool(x)
            x = self.conv2(x)
            return x, indices
        else:
            return self.conv(x)

class InvertedResidualBlockIndex(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidualBlockIndex, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        self.return_indices = expand_ratio != 1 and stride == 2

        if expand_ratio == 1:  # 3层以后都是6
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

        self.downsample1 = DepthwiseM2OIndexBlock(inp, use_nonlinear=True, use_context=True,batch_norm=nn.BatchNorm2d)
        self.conv1 = nn.Sequential(
            nn.Conv2d(oup, inp, 1, 1, 0,  bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0,  bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )
    def forward(self, x):
        if self.identity:  # 条件是步长为1，并且输入=输出
            return x + self.conv(x)
        elif self.return_indices:#条件是expand raio不是1，并且stride=2
            x = self.conv(x)
            x = self.conv1(x)
            idx1_en, idx1_de = self.downsample1(x)
            x = idx1_en * x
            x =  4 * F.avg_pool2d(x, (2, 2), stride=2)
            x = self.conv2(x)
            return x, idx1_de
        else:
            return self.conv(x)


class MobileNetV2_Encoder(nn.Module):
    def __init__(self):
        super(MobileNetV2_Encoder, self).__init__()
        self.enc_layer0 = ConvReLU6Block(3,32,3,2,1)
        self.enc_layer1= InvertedResidualBlock(32,16,1,1)
        self.enc_layer2= InvertedResidualBlock(16,24,2,6)
        self.enc_layer3= InvertedResidualBlock(24,24,1,6) 
        self.enc_layer4 = InvertedResidualBlock(24,32,2,6)
        self.enc_layer5 = InvertedResidualBlock(32,32,1,6)
        self.enc_layer6 = InvertedResidualBlock(32,32,1,6)
        self.enc_layer7 = InvertedResidualBlock(32,64,2,6)
        self.enc_layer8 = InvertedResidualBlock(64,64,1,6)
        self.enc_layer9 = InvertedResidualBlock(64,64,1,6)
        self.enc_layer10 = InvertedResidualBlock(64,64,1,6)
        self.enc_layer11 = InvertedResidualBlock(64,96,1,6)
        self.enc_layer12 = InvertedResidualBlock(96,96,1,6)
        self.enc_layer13 = InvertedResidualBlock(96,96,1,6)
        self.enc_layer14 = InvertedResidualBlock(96,160,2,6)
        self.enc_layer15 = InvertedResidualBlock(160,160,1,6)
        self.enc_layer16 = InvertedResidualBlock(160,160,1,6)
        self.enc_layer17 = InvertedResidualBlock(160,320,1,6)
        self.enc_layer18 = ConvReLU6Block(320,1280,1,1,0)
        self.enc_layer19 = nn.AvgPool2d(7)
        self.enc_output = nn.Linear(1280,1000)

    def forward(self, x):
        x=self.enc_layer0(x)
        x=self.enc_layer1(x)
        x=self.enc_layer2(x)
        x=self.enc_layer3(x)
        x=self.enc_layer4(x)
        x=self.enc_layer5(x)
        x=self.enc_layer6(x)
        x=self.enc_layer7(x)
        x=self.enc_layer8(x)
        x=self.enc_layer9(x)
        x=self.enc_layer10(x)
        x=self.enc_layer11(x)
        x=self.enc_layer12(x)
        x=self.enc_layer13(x)
        x=self.enc_layer14(x)
        x=self.enc_layer15(x)
        x=self.enc_layer16(x)
        x=self.enc_layer17(x)
        x=self.enc_layer18(x)
        x=self.enc_layer19(x)
        return self.enc_output(x)

class MobileNetV2_Encoder_pool(nn.Module):
    def __init__(self):
        super(MobileNetV2_Encoder_pool, self).__init__()
        self.enc_layer0 = ConvReLU6BlockPool(3,32,3,2,1)
        self.enc_layer1= InvertedResidualBlockPool(32,16,1,1)
        self.enc_layer2= InvertedResidualBlockPool(16,24,2,6)
        self.enc_layer3= InvertedResidualBlockPool(24,24,1,6)
        self.enc_layer4 = InvertedResidualBlockPool(24,32,2,6)
        self.enc_layer5 = InvertedResidualBlockPool(32,32,1,6)
        self.enc_layer6 = InvertedResidualBlockPool(32,32,1,6)
        self.enc_layer7 = InvertedResidualBlockPool(32,64,2,6)
        self.enc_layer8 = InvertedResidualBlockPool(64,64,1,6)
        self.enc_layer9 = InvertedResidualBlockPool(64,64,1,6)
        self.enc_layer10 = InvertedResidualBlockPool(64,64,1,6)
        self.enc_layer11 = InvertedResidualBlockPool(64,96,1,6)
        self.enc_layer12 = InvertedResidualBlockPool(96,96,1,6)
        self.enc_layer13 = InvertedResidualBlockPool(96,96,1,6)
        self.enc_layer14 = InvertedResidualBlockPool(96,160,2,6)
        self.enc_layer15 = InvertedResidualBlockPool(160,160,1,6)
        self.enc_layer16 = InvertedResidualBlockPool(160,160,1,6)
        self.enc_layer17 = InvertedResidualBlockPool(160,320,1,6)
        self.enc_layer18 = ConvReLU6Block(320,1280,1,1,0)
        self.enc_layer19 = nn.AvgPool2d(7)
        self.enc_output = nn.Linear(1280,1000)

    def forward(self, x):
        x=self.enc_layer0(x)
        x=self.enc_layer1(x)
        x=self.enc_layer2(x)
        x=self.enc_layer3(x)
        x=self.enc_layer4(x)
        x=self.enc_layer5(x)
        x=self.enc_layer6(x)
        x=self.enc_layer7(x)
        x=self.enc_layer8(x)
        x=self.enc_layer9(x)
        x=self.enc_layer10(x)
        x=self.enc_layer11(x)
        x=self.enc_layer12(x)
        x=self.enc_layer13(x)
        x=self.enc_layer14(x)
        x=self.enc_layer15(x)
        x=self.enc_layer16(x)
        x=self.enc_layer17(x)
        x=self.enc_layer18(x)
        x=self.enc_layer19(x)
        return self.enc_output(x)

class MobileNetV2_Encoder_carafe(nn.Module):
    def __init__(self):
        super(MobileNetV2_Encoder_carafe, self).__init__()
        self.enc_layer0 = ConvReLU6BlockCarafe(3,32,3,2,1)
        self.enc_layer1= InvertedResidualBlockCarafe(32,16,1,1)
        self.enc_layer2= InvertedResidualBlockCarafe(16,24,2,6)
        self.enc_layer3= InvertedResidualBlockCarafe(24,24,1,6)
        self.enc_layer4 = InvertedResidualBlockCarafe(24,32,2,6)
        self.enc_layer5 = InvertedResidualBlockCarafe(32,32,1,6)
        self.enc_layer6 = InvertedResidualBlockCarafe(32,32,1,6)
        self.enc_layer7 = InvertedResidualBlockCarafe(32,64,2,6)
        self.enc_layer8 = InvertedResidualBlockCarafe(64,64,1,6)
        self.enc_layer9 = InvertedResidualBlockCarafe(64,64,1,6)
        self.enc_layer10 = InvertedResidualBlockCarafe(64,64,1,6)
        self.enc_layer11 = InvertedResidualBlockCarafe(64,96,1,6)
        self.enc_layer12 = InvertedResidualBlockCarafe(96,96,1,6)
        self.enc_layer13 = InvertedResidualBlockCarafe(96,96,1,6)
        self.enc_layer14 = InvertedResidualBlockCarafe(96,160,2,6)
        self.enc_layer15 = InvertedResidualBlockCarafe(160,160,1,6)
        self.enc_layer16 = InvertedResidualBlockCarafe(160,160,1,6)
        self.enc_layer17 = InvertedResidualBlockCarafe(160,320,1,6)
        self.enc_layer18 = ConvReLU6Block(320,1280,1,1,0)
        self.enc_layer19 = nn.AvgPool2d(7)
        self.enc_output = nn.Linear(1280,1000)

    def forward(self, x):
        x=self.enc_layer0(x)
        x=self.enc_layer1(x)
        x=self.enc_layer2(x)
        x=self.enc_layer3(x)
        x=self.enc_layer4(x)
        x=self.enc_layer5(x)
        x=self.enc_layer6(x)
        x=self.enc_layer7(x)
        x=self.enc_layer8(x)
        x=self.enc_layer9(x)
        x=self.enc_layer10(x)
        x=self.enc_layer11(x)
        x=self.enc_layer12(x)
        x=self.enc_layer13(x)
        x=self.enc_layer14(x)
        x=self.enc_layer15(x)
        x=self.enc_layer16(x)
        x=self.enc_layer17(x)
        x=self.enc_layer18(x)
        x=self.enc_layer19(x)
        return self.enc_output(x)


class MobileNetV2_Encoder_dysample(nn.Module):
    def __init__(self):
        super(MobileNetV2_Encoder_dysample, self).__init__()
        self.enc_layer0 = ConvReLU6BlockDysample(3,32,3,2,1)
        self.enc_layer1= InvertedResidualBlockDysample(32,16,1,1)
        self.enc_layer2= InvertedResidualBlockDysample(16,24,2,6)
        self.enc_layer3= InvertedResidualBlockDysample(24,24,1,6)
        self.enc_layer4 = InvertedResidualBlockDysample(24,32,2,6)
        self.enc_layer5 = InvertedResidualBlockDysample(32,32,1,6)
        self.enc_layer6 = InvertedResidualBlockDysample(32,32,1,6)
        self.enc_layer7 = InvertedResidualBlockDysample(32,64,2,6)
        self.enc_layer8 = InvertedResidualBlockDysample(64,64,1,6)
        self.enc_layer9 = InvertedResidualBlockDysample(64,64,1,6)
        self.enc_layer10 = InvertedResidualBlockDysample(64,64,1,6)
        self.enc_layer11 = InvertedResidualBlockDysample(64,96,1,6)
        self.enc_layer12 = InvertedResidualBlockDysample(96,96,1,6)
        self.enc_layer13 = InvertedResidualBlockDysample(96,96,1,6)
        self.enc_layer14 = InvertedResidualBlockDysample(96,160,2,6)
        self.enc_layer15 = InvertedResidualBlockDysample(160,160,1,6)
        self.enc_layer16 = InvertedResidualBlockDysample(160,160,1,6)
        self.enc_layer17 = InvertedResidualBlockDysample(160,320,1,6)
        self.enc_layer18 = ConvReLU6Block(320,1280,1,1,0)
        self.enc_layer19 = nn.AvgPool2d(7)
        self.enc_output = nn.Linear(1280,1000)

    def forward(self, x):
        x=self.enc_layer0(x)
        x=self.enc_layer1(x)
        x=self.enc_layer2(x)
        x=self.enc_layer3(x)
        x=self.enc_layer4(x)
        x=self.enc_layer5(x)
        x=self.enc_layer6(x)
        x=self.enc_layer7(x)
        x=self.enc_layer8(x)
        x=self.enc_layer9(x)
        x=self.enc_layer10(x)
        x=self.enc_layer11(x)
        x=self.enc_layer12(x)
        x=self.enc_layer13(x)
        x=self.enc_layer14(x)
        x=self.enc_layer15(x)
        x=self.enc_layer16(x)
        x=self.enc_layer17(x)
        x=self.enc_layer18(x)
        x=self.enc_layer19(x)
        return self.enc_output(x)


class MobileNetV2_Encoder_s2d(nn.Module):
    def __init__(self):
        super(MobileNetV2_Encoder_s2d, self).__init__()
        self.enc_layer0 = ConvReLU6Blocks2d(3,32,3,2,1)
        self.enc_layer1= InvertedResidualBlockS2d(32,16,1,1)
        self.enc_layer2= InvertedResidualBlockS2d(16,24,2,6)
        self.enc_layer3= InvertedResidualBlockS2d(24,24,1,6)
        self.enc_layer4 = InvertedResidualBlockS2d(24,32,2,6)
        self.enc_layer5 = InvertedResidualBlockS2d(32,32,1,6)
        self.enc_layer6 = InvertedResidualBlockS2d(32,32,1,6)
        self.enc_layer7 = InvertedResidualBlockS2d(32,64,2,6)
        self.enc_layer8 = InvertedResidualBlockS2d(64,64,1,6)
        self.enc_layer9 = InvertedResidualBlockS2d(64,64,1,6)
        self.enc_layer10 = InvertedResidualBlockS2d(64,64,1,6)
        self.enc_layer11 = InvertedResidualBlockS2d(64,96,1,6)
        self.enc_layer12 = InvertedResidualBlockS2d(96,96,1,6)
        self.enc_layer13 = InvertedResidualBlockS2d(96,96,1,6)
        self.enc_layer14 = InvertedResidualBlockS2d(96,160,2,6)
        self.enc_layer15 = InvertedResidualBlockS2d(160,160,1,6)
        self.enc_layer16 = InvertedResidualBlockS2d(160,160,1,6)
        self.enc_layer17 = InvertedResidualBlockS2d(160,320,1,6)
        self.enc_layer18 = ConvReLU6Block(320,1280,1,1,0)
        self.enc_layer19 = nn.AvgPool2d(7)
        self.enc_output = nn.Linear(1280,1000)

    def forward(self, x):
        x=self.enc_layer0(x)
        x=self.enc_layer1(x)
        x=self.enc_layer2(x)
        x=self.enc_layer3(x)
        x=self.enc_layer4(x)
        x=self.enc_layer5(x)
        x=self.enc_layer6(x)
        x=self.enc_layer7(x)
        x=self.enc_layer8(x)
        x=self.enc_layer9(x)
        x=self.enc_layer10(x)
        x=self.enc_layer11(x)
        x=self.enc_layer12(x)
        x=self.enc_layer13(x)
        x=self.enc_layer14(x)
        x=self.enc_layer15(x)
        x=self.enc_layer16(x)
        x=self.enc_layer17(x)
        x=self.enc_layer18(x)
        x=self.enc_layer19(x)
        return self.enc_output(x)

class MobileNetV2_Encoder_indexed(nn.Module):
    def __init__(self):
        super(MobileNetV2_Encoder_indexed, self).__init__()
        self.enc_layer0 = ConvReLU6Blockindexed(3,32,3,2,1)
        self.enc_layer1= InvertedResidualBlockIndex(32,16,1,1)
        self.enc_layer2= InvertedResidualBlockIndex(16,24,2,6)
        self.enc_layer3= InvertedResidualBlockIndex(24,24,1,6)
        self.enc_layer4 = InvertedResidualBlockIndex(24,32,2,6)
        self.enc_layer5 = InvertedResidualBlockIndex(32,32,1,6)
        self.enc_layer6 = InvertedResidualBlockIndex(32,32,1,6)
        self.enc_layer7 = InvertedResidualBlockIndex(32,64,2,6)
        self.enc_layer8 = InvertedResidualBlockIndex(64,64,1,6)
        self.enc_layer9 = InvertedResidualBlockIndex(64,64,1,6)
        self.enc_layer10 = InvertedResidualBlockIndex(64,64,1,6)
        self.enc_layer11 = InvertedResidualBlockIndex(64,96,1,6)
        self.enc_layer12 = InvertedResidualBlockIndex(96,96,1,6)
        self.enc_layer13 = InvertedResidualBlockIndex(96,96,1,6)
        self.enc_layer14 = InvertedResidualBlockIndex(96,160,2,6)
        self.enc_layer15 = InvertedResidualBlockIndex(160,160,1,6)
        self.enc_layer16 = InvertedResidualBlockIndex(160,160,1,6)
        self.enc_layer17 = InvertedResidualBlockIndex(160,320,1,6)
        self.enc_layer18 = ConvReLU6Block(320,1280,1,1,0)
        self.enc_layer19 = nn.AvgPool2d(7)
        self.enc_output = nn.Linear(1280,1000)

    def forward(self, x):
        x=self.enc_layer0(x)
        x=self.enc_layer1(x)
        x=self.enc_layer2(x)
        x=self.enc_layer3(x)
        x=self.enc_layer4(x)
        x=self.enc_layer5(x)
        x=self.enc_layer6(x)
        x=self.enc_layer7(x)
        x=self.enc_layer8(x)
        x=self.enc_layer9(x)
        x=self.enc_layer10(x)
        x=self.enc_layer11(x)
        x=self.enc_layer12(x)
        x=self.enc_layer13(x)
        x=self.enc_layer14(x)
        x=self.enc_layer15(x)
        x=self.enc_layer16(x)
        x=self.enc_layer17(x)
        x=self.enc_layer18(x)
        x=self.enc_layer19(x)
        return self.enc_output(x)


class NNConv5_DecoderV2(nn.Module):          #卷积核大小为5
  def __init__(self, kernel_size, depthwise=True):
    super(NNConv5_DecoderV2, self).__init__()
    if (depthwise):
      self.conv1 = nn.Sequential(DWConvBlock(320,320,kernel_size,1),ConvBlock(320,96,1,1,0)) #14X14
      self.conv2 = nn.Sequential(DWConvBlock(96,96,kernel_size,1),ConvBlock(96,32,1,1,0)) #28 X 28
      self.conv3 = nn.Sequential(DWConvBlock(32,32,kernel_size,1),ConvBlock(32,24,1,1,0)) # 56X56
      self.conv4 = nn.Sequential(DWConvBlock(24,24,kernel_size,1),ConvBlock(24,16,1,1,0)) #112 X 112
      self.conv5 = nn.Sequential(DWConvBlock(16,16,kernel_size,1),ConvBlock(16,32,1,1,0)) #224 X 224

    self.output = ConvBlock(32,1,1,1,0)
  def forward(self,x):
    x = F.interpolate(self.conv1(x), scale_factor=2, mode='bilinear',align_corners=True)
    x = F.interpolate(self.conv2(x), scale_factor=2, mode='bilinear',align_corners=True)
    x = F.interpolate(self.conv3(x), scale_factor=2, mode='bilinear',align_corners=True)
    x = F.interpolate(self.conv4(x), scale_factor=2, mode='bilinear',align_corners=True)
    x = F.interpolate(self.conv5(x), scale_factor=2, mode='bilinear',align_corners=True)
    return self.output(x)

class FastDepthV2(nn.Module):
  def __init__(self,kernel_size=5):
    super(FastDepthV2,self).__init__()
    if my_choice == 'conv-bilinear' or my_choice == 'conv-deconv'or my_choice == 'upsample_only':
        self.encoder= MobileNetV2_Encoder()
        self.decoder = NNConv5_DecoderV2(kernel_size)
    elif my_choice == 'max_pooling-max_unpooling':
        self.encoder = MobileNetV2_Encoder_pool()
        self.decoder = NNConv5_DecoderV2(kernel_size)
    elif my_choice == 'carafe-carafe++':
        self.encoder = MobileNetV2_Encoder_carafe()
        self.decoder = NNConv5_DecoderV2(kernel_size)
    elif my_choice == 'co_dysample':
        self.encoder = MobileNetV2_Encoder_dysample()
        self.decoder = NNConv5_DecoderV2(kernel_size)
    elif my_choice == 'space2depth-depth2space':
        self.encoder = MobileNetV2_Encoder_s2d()
        self.decoder = NNConv5_DecoderV2(kernel_size)
    elif my_choice == 'indexed':
        self.encoder = MobileNetV2_Encoder_indexed()
        self.decoder = NNConv5_DecoderV2(kernel_size)
  def forward(self,x):
      #现在是下采样部分
    if my_choice == 'conv-bilinear' or my_choice == 'upsample_only' or my_choice == 'space2depth-depth2space' or my_choice == 'conv-deconv' or my_choice == 'carafe-carafe++' :
        #刚进入时   x = 8,3,224,224
        x=self.encoder.enc_layer0(x)# x = 8,32,112,112
        x=self.encoder.enc_layer1(x)# x = 8,16,112,112
        layer1= x                   #layer1 = 8,16,112,112
        x=self.encoder.enc_layer2(x)# x = 8,24,56,56
        x=self.encoder.enc_layer3(x)# x = 8,24,56,56
        layer2=x  #layer2 = 8,24,56,56
        x=self.encoder.enc_layer4(x)#执行完此行以后 x = 8,32 28 28
        x=self.encoder.enc_layer5(x)#执行完此行以后 x = 8,32 28 28
        layer3=x                    #layer3 = 8,32 28 28
        x=self.encoder.enc_layer6(x)#执行完此行以后 x = 8,32 28 28
        x=self.encoder.enc_layer7(x)#执行完此行以后 x = 8,64,14,14
        x=self.encoder.enc_layer8(x)#执行完此行以后 x = 8,64,14,14
        x=self.encoder.enc_layer9(x)#执行完此行以后 x = 8,64,14,14
        x=self.encoder.enc_layer10(x)#执行完此行以后 x = 8,64,14,14
        x=self.encoder.enc_layer11(x)#执行完此行以后 x = 8,96,14,14
        x=self.encoder.enc_layer12(x)#执行完此行以后 x = 8,96,14,14
        x=self.encoder.enc_layer13(x)#执行完此行以后 x = 8,96,14,14
        x=self.encoder.enc_layer14(x)#执行完此行以后 x = 8,160,7,7
        x=self.encoder.enc_layer15(x)#执行完此行以后 x = 8,160,7,7
        x=self.encoder.enc_layer16(x)#执行完此行以后 x = 8,160,7,7
        x= self.encoder.enc_layer17(x)#执行完此行以后 x = 8,320,7,7
    elif my_choice == 'max_pooling-max_unpooling':        # 刚进入时   x = 8,3,224,224
        x,pool_indices[0] = self.encoder.enc_layer0(x)  # x = 8,32,112,112
        x = self.encoder.enc_layer1(x)  # x = 8,16,112,112
        layer1 = x  # layer1 = 8,16,112,112
        x ,pool_indices[1]= self.encoder.enc_layer2(x)  # x = 8,24,56,56
        x = self.encoder.enc_layer3(x)  # x = 8,24,56,56
        layer2 = x  # layer2 = 8,24,56,56
        x ,pool_indices[2] = self.encoder.enc_layer4(x)  # 执行完此行以后 x = 8,32 28 28
        x = self.encoder.enc_layer5(x)  # 执行完此行以后 x = 8,32 28 28
        layer3 = x  # layer3 = 8,32 28 28
        x = self.encoder.enc_layer6(x)  # 执行完此行以后 x = 8,32 28 28
        x ,pool_indices[3] = self.encoder.enc_layer7(x)  # 执行完此行以后 x = 8,64,14,14
        x = self.encoder.enc_layer8(x)  # 执行完此行以后 x = 8,64,14,14
        x = self.encoder.enc_layer9(x)  # 执行完此行以后 x = 8,64,14,14
        x = self.encoder.enc_layer10(x)  # 执行完此行以后 x = 8,64,14,14
        x = self.encoder.enc_layer11(x)  # 执行完此行以后 x = 8,96,14,14
        x = self.encoder.enc_layer12(x)  # 执行完此行以后 x = 8,96,14,14
        x = self.encoder.enc_layer13(x)  # 执行完此行以后 x = 8,96,14,14
        x ,pool_indices[4] = self.encoder.enc_layer14(x)  # 执行完此行以后 x = 8,160,7,7
        x = self.encoder.enc_layer15(x)  # 执行完此行以后 x = 8,160,7,7
        x = self.encoder.enc_layer16(x)  # 执行完此行以后 x = 8,160,7,7
        x = self.encoder.enc_layer17(x)  # 执行完此行以后 x = 8,320,7,7
    elif my_choice == 'co_dysample':
        # 刚进入时   x = 8,3,224,224
        x,offset[0] = self.encoder.enc_layer0(x)  # x = 8,32,112,112
        x = self.encoder.enc_layer1(x)  # x = 8,16,112,112
        layer1 = x  # layer1 = 8,16,112,112
        x,offset[1] = self.encoder.enc_layer2(x)  # x = 8,24,56,56
        x = self.encoder.enc_layer3(x)  # x = 8,24,56,56
        layer2 = x  # layer2 = 8,24,56,56
        x ,offset[2]= self.encoder.enc_layer4(x)  # 执行完此行以后 x = 8,32 28 28
        x = self.encoder.enc_layer5(x)  # 执行完此行以后 x = 8,32 28 28
        layer3 = x  # layer3 = 8,32 28 28
        x = self.encoder.enc_layer6(x)  # 执行完此行以后 x = 8,32 28 28
        x ,offset[3]= self.encoder.enc_layer7(x)  # 执行完此行以后 x = 8,64,14,14
        x = self.encoder.enc_layer8(x)  # 执行完此行以后 x = 8,64,14,14
        x = self.encoder.enc_layer9(x)  # 执行完此行以后 x = 8,64,14,14
        x = self.encoder.enc_layer10(x)  # 执行完此行以后 x = 8,64,14,14
        x = self.encoder.enc_layer11(x)  # 执行完此行以后 x = 8,96,14,14
        x = self.encoder.enc_layer12(x)  # 执行完此行以后 x = 8,96,14,14
        x = self.encoder.enc_layer13(x)  # 执行完此行以后 x = 8,96,14,14
        x ,offset[4]= self.encoder.enc_layer14(x)  # 执行完此行以后 x = 8,160,7,7
        x = self.encoder.enc_layer15(x)  # 执行完此行以后 x = 8,160,7,7
        x = self.encoder.enc_layer16(x)  # 执行完此行以后 x = 8,160,7,7
        x = self.encoder.enc_layer17(x)  # 执行完此行以后 x = 8,320,7,7
    elif my_choice == 'indexed':
        # 刚进入时   x = 8,3,224,224
        x, index_decoder[0] = self.encoder.enc_layer0(x)  # x = 8,32,112,112
        x = self.encoder.enc_layer1(x)  # x = 8,16,112,112
        layer1 = x  # layer1 = 8,16,112,112
        x, index_decoder[1] = self.encoder.enc_layer2(x)  # x = 8,24,56,56
        x = self.encoder.enc_layer3(x)  # x = 8,24,56,56
        layer2 = x  # layer2 = 8,24,56,56
        x, index_decoder[2] = self.encoder.enc_layer4(x)  # 执行完此行以后 x = 8,32 28 28
        x = self.encoder.enc_layer5(x)  # 执行完此行以后 x = 8,32 28 28
        layer3 = x  # layer3 = 8,32 28 28
        x = self.encoder.enc_layer6(x)  # 执行完此行以后 x = 8,32 28 28
        x, index_decoder[3] = self.encoder.enc_layer7(x)  # 执行完此行以后 x = 8,64,14,14
        x = self.encoder.enc_layer8(x)  # 执行完此行以后 x = 8,64,14,14
        x = self.encoder.enc_layer9(x)  # 执行完此行以后 x = 8,64,14,14
        x = self.encoder.enc_layer10(x)  # 执行完此行以后 x = 8,64,14,14
        x = self.encoder.enc_layer11(x)  # 执行完此行以后 x = 8,96,14,14
        x = self.encoder.enc_layer12(x)  # 执行完此行以后 x = 8,96,14,14
        x = self.encoder.enc_layer13(x)  # 执行完此行以后 x = 8,96,14,14
        x, index_decoder[4] = self.encoder.enc_layer14(x)  # 执行完此行以后 x = 8,160,7,7
        x = self.encoder.enc_layer15(x)  # 执行完此行以后 x = 8,160,7,7
        x = self.encoder.enc_layer16(x)  # 执行完此行以后 x = 8,160,7,7
        x = self.encoder.enc_layer17(x)  # 执行完此行以后 x = 8,320,7,7
      #现在是上采样
    if my_choice == 'conv-bilinear':
        x=self.decoder.conv1(x)
        x= F.interpolate(x, scale_factor=2, mode='bilinear',align_corners=True)    #执行完此行以后 x = 8,96,14,14
        x=self.decoder.conv2(x)
        x= F.interpolate(x, scale_factor=2, mode='bilinear',align_corners=True)    #执行完此行以后 x = 8,32 28 28
        x = x+layer3
        x=self.decoder.conv3(x)
        x= F.interpolate(x, scale_factor=2, mode='bilinear',align_corners=True)    #执行完此行以后 x = 8,24,56,56
        x = x+layer2
        x= F.interpolate(self.decoder.conv4(x), scale_factor=2, mode='bilinear',align_corners=True)    #执行完此行以后 x = 8,16,112,112
        x = x+layer1
        x= F.interpolate(self.decoder.conv5(x), scale_factor=2, mode='bilinear',align_corners=True)    #执行完此行以后 x = 8,32,224,224
        # x = F.interpolate(self.decoder.conv1(x), scale_factor=2, mode='nearest',
        #                   )  # 执行完此行以后 x = 8,96,14,14
        # x = F.interpolate(self.decoder.conv2(x), scale_factor=2, mode='nearest',
        #                   )  # 执行完此行以后 x = 8,32 28 28
        # x = x + layer3
        # x = F.interpolate(self.decoder.conv3(x), scale_factor=2, mode='nearest',
        #                   )  # 执行完此行以后 x = 8,24,56,56
        # x = x + layer2
        # x = F.interpolate(self.decoder.conv4(x), scale_factor=2, mode='nearest',
        #                   )  # 执行完此行以后 x = 8,16,112,112
        # x = x + layer1
        # x = F.interpolate(self.decoder.conv5(x), scale_factor=2, mode='nearest')
        return self.decoder.output(x)
    elif my_choice == 'conv-deconv':
        x = self.decoder.conv1(x)
        deconv_layer1 = DeconvBlock(
            in_channels=96,
            out_channels=96,kernel_size=3
        )
        deconv_layer1=deconv_layer1.cuda()
        x=deconv_layer1(x)
        x = self.decoder.conv2(x)
        deconv_layer2 = DeconvBlock(
            in_channels=32,
            out_channels=32,kernel_size=3
        )
        deconv_layer2=deconv_layer2.cuda()
        x=deconv_layer2(x)
        x = x + layer3
        x = self.decoder.conv3(x)
        deconv_layer3 = DeconvBlock(
            in_channels=24,
            out_channels=24,kernel_size=3)
        deconv_layer3=deconv_layer3.cuda()
        x=deconv_layer3(x)
        x = x + layer2
        x = self.decoder.conv4(x)
        deconv_layer4 = DeconvBlock(
            in_channels=16,
            out_channels=16,kernel_size=3)
        deconv_layer4=deconv_layer4.cuda()
        x = deconv_layer4(x)
        x = x + layer1
        x = self.decoder.conv5(x)
        deconv_layer5 = DeconvBlock(
            in_channels=32,
            out_channels=32,
            kernel_size=2)
        deconv_layer5=deconv_layer5.cuda()
        x = deconv_layer5(x)
        return self.decoder.output(x)
      # x = self.decoder.conv1(x)
      # deconv_layer1 = nn.ConvTranspose2d(
      #     in_channels=96,
      #     out_channels=96, kernel_size=3, padding=1, stride=2, output_padding=1, bias=True
      # )
      # deconv_layer1 = deconv_layer1.cuda()
      # x = deconv_layer1(x)
      # x = self.decoder.conv2(x)
      # deconv_layer2 = nn.ConvTranspose2d(
      #     in_channels=32,
      #     out_channels=32, kernel_size=3, padding=1, stride=2, output_padding=1, bias=True
      # )
      # deconv_layer2 = deconv_layer2.cuda()
      # x = deconv_layer2(x)
      # x = x + layer3
      # x = self.decoder.conv3(x)
      # deconv_layer3 = nn.ConvTranspose2d(
      #     in_channels=24,
      #     out_channels=24, kernel_size=3, padding=1, stride=2, output_padding=1, bias=True)
      # deconv_layer3 = deconv_layer3.cuda()
      # x = deconv_layer3(x)
      # x = x + layer2
      # x = self.decoder.conv4(x)
      # deconv_layer4 = nn.ConvTranspose2d(
      #     in_channels=16,
      #     out_channels=16, kernel_size=3, padding=1, stride=2, output_padding=1, bias=True)
      # deconv_layer4 = deconv_layer4.cuda()
      # x = deconv_layer4(x)
      # x = x + layer1
      # x = self.decoder.conv5(x)
      # deconv_layer5 = nn.ConvTranspose2d(
      #     in_channels=32,
      #     out_channels=32,
      #     kernel_size=2,
      #     stride=2,
      #     padding=0,
      #     output_padding=0,
      #     bias=True)
      # deconv_layer5 = deconv_layer5.cuda()
      # x = deconv_layer5(x)
      # return self.decoder.output(x)
    elif my_choice == 'max_pooling-max_unpooling':
        # ini0=pool_indices[0]
        # ini1=pool_indices[1]
        # ini2=pool_indices[2]
        # ini3=pool_indices[3]
        # ini4=pool_indices[4]
        x = self.decoder.conv1(x)
        unpool_layer1 = nn.MaxUnpool2d(2,2)
        unpool_layer1 = unpool_layer1.cuda()
        x = unpool_layer1(x, pool_indices[4])
        x = self.decoder.conv2(x)
        x = unpool_layer1(x, pool_indices[3])
        x = x + layer3
        x = self.decoder.conv3(x)
        x = unpool_layer1(x, pool_indices[2])
        x = x + layer2
        x = self.decoder.conv4(x)
        x = unpool_layer1(x, pool_indices[1])
        x = x + layer1
        x = self.decoder.conv5(x)
        x = unpool_layer1(x, pool_indices[0])
        return self.decoder.output(x)
    elif my_choice == 'co_dysample':
        x = self.decoder.conv1(x)
        dysample1 = Dy_UpSample(96, 'lp', 2, 4, True)
        dysample1 = dysample1.cuda()
        x = dysample1(x,offset[4])

        x = self.decoder.conv2(x)
        dysample2 = Dy_UpSample(32, 'lp', 2, 4, True)
        dysample2 = dysample2.cuda()
        x = dysample2(x,offset[3])

        x = x + layer3

        x = self.decoder.conv3(x)
        dysample3 = Dy_UpSample(24, 'lp', 2, 4, True)
        dysample3 = dysample3.cuda()
        x = dysample3(x,offset[2])

        x = x + layer2
        x = self.decoder.conv4(x)
        dysample4 = Dy_UpSample(16, 'lp', 2, 4, True)
        dysample4 = dysample4.cuda()
        x = dysample4(x,offset[1])
        x = x + layer1
        x = self.decoder.conv5(x)
        dysample5 = Dy_UpSample(32, 'lp', 2, 4, True)
        dysample5 = dysample5.cuda()
        x = dysample5(x,offset[0])
        return self.decoder.output(x)
    elif my_choice == 'space2depth-depth2space':
        x = self.decoder.conv1(x)
        p1 = nn.PixelShuffle(2)
        p1 = p1.cuda()
        x = p1(x)
        channel_num = x.shape[1]
        restore_layer = nn.Conv2d(in_channels=channel_num, out_channels=4*channel_num, kernel_size=1, padding=0, stride=1, bias=False)
        restore_layer = restore_layer.cuda()
        x = restore_layer(x)
        x = self.decoder.conv2(x)
        x = p1(x)
        channel_num = x.shape[1]
        restore_layer = nn.Conv2d(in_channels=channel_num, out_channels=4*channel_num, kernel_size=1, padding=0, stride=1, bias=False)
        restore_layer = restore_layer.cuda()
        x = restore_layer(x)

        x = x + layer3
        x = self.decoder.conv3(x)
        x = p1(x)
        channel_num = x.shape[1]
        restore_layer = nn.Conv2d(in_channels=channel_num, out_channels=4*channel_num, kernel_size=1, padding=0, stride=1, bias=False)
        restore_layer = restore_layer.cuda()
        x = restore_layer(x)

        x = x + layer2
        x = self.decoder.conv4(x)
        x = p1(x)
        channel_num = x.shape[1]
        restore_layer = nn.Conv2d(in_channels=channel_num, out_channels=4*channel_num, kernel_size=1, padding=0, stride=1, bias=False)
        restore_layer = restore_layer.cuda()
        x = restore_layer(x)
        x = x + layer1
        x = self.decoder.conv5(x)
        x = p1(x)
        channel_num = x.shape[1]
        restore_layer = nn.Conv2d(in_channels=channel_num, out_channels=4 * channel_num, kernel_size=1, padding=0,
                                  stride=1, bias=False)
        restore_layer = restore_layer.cuda()
        x = restore_layer(x)

        return self.decoder.output(x)
    elif my_choice == 'indexed':
        x = self.decoder.conv1(x)
        a1=index_decoder[4]# 断点调试用
        a2=index_decoder[3]

        x = index_decoder[4] * F.interpolate(x,scale_factor=2, mode='nearest')# 上采样尺寸扩大两倍
        x = self.decoder.conv2(x)
        x = index_decoder[3] * F.interpolate(x,scale_factor=2, mode='nearest')# 上采样尺寸扩大两倍
        x = x + layer3
        x = self.decoder.conv3(x)
        x = index_decoder[2] * F.interpolate(x,scale_factor=2, mode='nearest')# 上采样尺寸扩大两倍
        x = x + layer2
        x = self.decoder.conv4(x)
        x = index_decoder[1] * F.interpolate(x, scale_factor=2, mode='nearest')# 上采样尺寸扩大两倍
        x = x + layer1
        x = self.decoder.conv5(x)
        x = index_decoder[0] * F.interpolate(x, scale_factor=2, mode='nearest')# 上采样尺寸扩大两倍
        return self.decoder.output(x)
    elif my_choice == 'carafe-carafe++' or my_choice == 'upsample_only':
        x = self.decoder.conv1(x)
        carafe1 = CARAFE_upsample(96, 64, 5, ratio=2)
        carafe1 = carafe1.cuda()
        x = carafe1(x)


        x = self.decoder.conv2(x)
        carafe2 = CARAFE_upsample(32, 64, 5, ratio=2)
        carafe2 = carafe2.cuda()
        x = carafe2(x)
        x = x + layer3
        x = self.decoder.conv3(x)
        carafe3 = CARAFE_upsample(24, 64, 5, ratio=2)
        carafe3 = carafe3.cuda()
        x = carafe3(x)
        x = x + layer2
        x = self.decoder.conv4(x)
        carafe4 = CARAFE_upsample(16, 64, 5, ratio=2)
        carafe4 = carafe4.cuda()
        x = carafe4(x)
        x = x + layer1
        x = self.decoder.conv5(x)
        carafe5 = CARAFE_upsample(32, 64, 5, ratio=2)
        carafe5 = carafe5.cuda()
        x = carafe5(x)
        return self.decoder.output(x)

