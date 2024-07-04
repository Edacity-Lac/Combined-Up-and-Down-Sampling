def load_network_structure():# 从argparse模块获得网络上下采样的方法
    args = utils.parse_args()
    return args.encoder_decoder_choice

my_choice = load_network_structure() # 从argparse模块获得网络上下采样的方法

def DeconvBlock(in_channels,out_channels,kernel_size):# 此方法为解码器中采用的模块之一
  return nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size,
                stride=2,padding = (kernel_size - 1) // 2,output_padding = kernel_size%2,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
def ConvBlock(in_channels,out_channels,kernel_size,stride,padding):# 此方法为解码器中采用的模块之一
  return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
class ConvReLU6Blockindexed(nn.Module): # 此类为编码器中第1层采用的模块
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


class InvertedResidualBlockIndex(nn.Module): # 此类为编码器中第2-17层采用的模块
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
class MobileNetV2_Encoder_indexed(nn.Module):#此类为index下采样方法使用的编码器
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



class NNConv5_DecoderV2(nn.Module):          #此类为index上采样方法使用的解码器
  def __init__(self, kernel_size, depthwise=True):
    super(NNConv5_DecoderV2, self).__init__()
    if (depthwise):
      self.conv1 = nn.Sequential(DWConvBlock(320,320,kernel_size,1),ConvBlock(320,96,1,1,0)) #14X14
      self.conv2 = nn.Sequential(DWConvBlock(96,96,kernel_size,1),ConvBlock(96,32,1,1,0)) #28 X 28
      self.conv3 = nn.Sequential(DWConvBlock(32,32,kernel_size,1),ConvBlock(32,24,1,1,0)) # 56X56
      self.conv4 = nn.Sequential(DWConvBlock(24,24,kernel_size,1),ConvBlock(24,16,1,1,0)) #112 X 112
      self.conv5 = nn.Sequential(DWConvBlock(16,16,kernel_size,1),ConvBlock(16,32,1,1,0)) #224 X 224
    self.output = ConvBlock(32,1,1,1,0)

class FastDepthV2(nn.Module): # 此类描述了不同算子的网络核心架构，现以index方法为例进行展示
  def __init__(self,kernel_size=5):
    super(FastDepthV2,self).__init__()
    if my_choice == 'indexed':
        self.encoder = MobileNetV2_Encoder_indexed()
        self.decoder = NNConv5_DecoderV2(kernel_size)
    #如果是其他的上下采样算子，采取其他的编码器解码器。

  def forward(self,x):
      #现在是下采样部分
      if my_choice == 'indexed':  # index下采样进入这个分支判断。如果是别的下采样方法，进入别的判断。
          # 刚进入时   x = 8,3,224,224
          x, index_decoder[0] = self.encoder.enc_layer0(x)  # x = 8,32,112,112
          x = self.encoder.enc_layer1(x)  # x = 8,16,112,112
          layer1 = x  # layer1 = 8,16,112,112
          x, index_decoder[1] = self.encoder.enc_layer2(x)  # x = 8,24,56,56
          x = self.encoder.enc_layer3(x)  # x = 8,24,56,56
          layer2 = x  # layer2 = 8,24,56,56
          x, index_decoder[2] = self.encoder.enc_layer4(x)  # x = 8,32 28 28
          x = self.encoder.enc_layer5(x)  # x = 8,32 28 28
          layer3 = x  # layer3 = 8,32 28 28
          x = self.encoder.enc_layer6(x)  # x = 8,32 28 28
          x, index_decoder[3] = self.encoder.enc_layer7(x)  # x = 8,64,14,14
          x = self.encoder.enc_layer8(x)  # x = 8,64,14,14
          x = self.encoder.enc_layer9(x)  # x = 8,64,14,14
          x = self.encoder.enc_layer10(x)  # x = 8,64,14,14
          x = self.encoder.enc_layer11(x)  # x = 8,96,14,14
          x = self.encoder.enc_layer12(x)  # x = 8,96,14,14
          x = self.encoder.enc_layer13(x)  # x = 8,96,14,14
          x, index_decoder[4] = self.encoder.enc_layer14(x)  # x = 8,160,7,7
          x = self.encoder.enc_layer15(x)  # x = 8,160,7,7
          x = self.encoder.enc_layer16(x)  # x = 8,160,7,7
          x = self.encoder.enc_layer17(x)  # x = 8,320,7,7
     # 现在是上采样
      elif my_choice == 'indexed': # index上采样进入这个分支判断。如果是别的上采样方法，进入别的判断。
          x = self.decoder.conv1(x)
          x = index_decoder[4] * F.interpolate(x, scale_factor=2, mode='nearest')  # x = 8,96,14,14
          x = self.decoder.conv2(x)
          x = index_decoder[3] * F.interpolate(x, scale_factor=2, mode='nearest')  # x = 8,32 28 28
          x = x + layer3
          x = self.decoder.conv3(x)
          x = index_decoder[2] * F.interpolate(x, scale_factor=2, mode='nearest')  # x = 8,24,56,56
          x = x + layer2
          x = self.decoder.conv4(x)
          x = index_decoder[1] * F.interpolate(x, scale_factor=2, mode='nearest')  # x = 8,16,112,112
          x = x + layer1
          x = self.decoder.conv5(x)
          x = index_decoder[0] * F.interpolate(x, scale_factor=2, mode='nearest')  # x = 8,32,224,224
          return self.decoder.output(x)