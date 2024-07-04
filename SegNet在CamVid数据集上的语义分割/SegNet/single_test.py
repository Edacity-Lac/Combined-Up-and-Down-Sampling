'''
Use:
conda activate segnet && \
python single_test.py \
    --image_path test/0001TP_008550.png \
    --output_path test/result_0001TP_008550.png \
    --model SegNet \
    --checkpoint checkpoint/camvid/SegNetbs8gpu1_train/model_200.pth \
    --cuda \
    --gpus 0 \
    --num_classes 11
    --num_workers 8
'''
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
from PIL import Image
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_test
from utils.convert_state import convert_state_dict


# 颜色映射表
'''CAMVID_CLASSES = ['Sky',
                  'Building',
                  'Pole',
                  'Road',
                  'Sidewalk',
                  'Tree',
                  'SignSymbol',
                  'Fence',
                  'Car',
                  'Pedestrian',
                  'Bicyclist',
                  'Void']
'''
color_map = {
    0: [128, 128, 128],
    1: [128, 0, 0],
    2: [192, 192, 128],
    3: [128, 64, 128],
    4: [0, 0, 192],
    5: [128, 128, 0],
    6: [192, 128, 128],
    7: [64, 64, 128],
    8: [64, 0, 128],
    9: [64, 64, 0],
    10: [0, 128, 192],
    11: [0, 0, 0],
}

    
class HalfModel(nn.Module):
    def __init__(self, original_model):
        super(HalfModel, self).__init__()
        self.original_model = original_model

    def forward(self, x):
         # Stage 1
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        # x1_size = x.size()
        #x1p, id1 = F.max_pool2d(x12,kernel_size=2, stride=2,return_indices=True)
        # x1p=self.downsample1(x12)
        x=self.downsample1(x)

        # Stage 2
        # x21 = F.relu(self.bn21(self.conv21(x1p)))
        x = F.relu(self.bn21(self.conv21(x)))
        x = F.relu(self.bn22(self.conv22(x)))
        #x2_size = x.size()
        # x2p, id2 = F.max_pool2d(x22,kernel_size=2, stride=2,return_indices=True)
        #x, id2 = F.max_pool2d(x,kernel_size=2, stride=2,return_indices=True)
        x=self.downsample2(x)

        # Stage 3
        # x31 = F.relu(self.bn31(self.conv31(x2p)))
        x = F.relu(self.bn31(self.conv31(x)))
        x = F.relu(self.bn32(self.conv32(x)))
        x = F.relu(self.bn33(self.conv33(x)))
        #x3_size = x.size()
        # x3p, id3 = F.max_pool2d(x33,kernel_size=2, stride=2,return_indices=True)
        #x, id3 = F.max_pool2d(x,kernel_size=2, stride=2,return_indices=True)
        x=self.downsample3(x)

        # Stage 4
        # x41 = F.relu(self.bn41(self.conv41(x3p)))
        x = F.relu(self.bn41(self.conv41(x)))
        x = F.relu(self.bn42(self.conv42(x)))
        x = F.relu(self.bn43(self.conv43(x)))
        #x4_size = x.size()
        # x4p, id4 = F.max_pool2d(x43,kernel_size=2, stride=2,return_indices=True)
        #x, id4 = F.max_pool2d(x,kernel_size=2, stride=2,return_indices=True)
        x=self.downsample4(x)

        # Stage 5
        # x51 = F.relu(self.bn51(self.conv51(x4p)))
        x = F.relu(self.bn51(self.conv51(x)))
        x = F.relu(self.bn52(self.conv52(x)))
        x = F.relu(self.bn53(self.conv53(x)))
        #x5_size = x.size()
        # x5p, id5 = F.max_pool2d(x53,kernel_size=2, stride=2,return_indices=True)
        #x, id5 = F.max_pool2d(x,kernel_size=2, stride=2,return_indices=True)
        x=self.downsample5(x)


        # Stage 5d
        # x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2, output_size=x5_size)
        #x = F.max_unpool2d(x, id5, kernel_size=2, stride=2, output_size=x5_size)
        x = self.upsample1(x)
        # x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x5 = F.relu(self.bn53d(self.conv53d(x)))
        x5 = F.relu(self.bn52d(self.conv52d(x5)))
        x = F.relu(self.bn51d(self.conv51d(x)))

        # Stage 4d
        # x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2, output_size=x4_size)
        #x = F.max_unpool2d(x, id4, kernel_size=2, stride=2, output_size=x4_size)
        x = self.upsample2(x)
        x = F.interpolate(x, size=(45, 60), mode='bilinear', align_corners=False) #填充为45*60
        # x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x = F.relu(self.bn43d(self.conv43d(x)))
        x = F.relu(self.bn42d(self.conv42d(x)))
        x = F.relu(self.bn41d(self.conv41d(x)))

        # Stage 3d
        # x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2, output_size=x3_size)
        #x = F.max_unpool2d(x, id3, kernel_size=2, stride=2, output_size=x3_size)
        x = self.upsample3(x)
        # x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x = F.relu(self.bn33d(self.conv33d(x)))
        x = F.relu(self.bn32d(self.conv32d(x)))
        x = F.relu(self.bn31d(self.conv31d(x)))

        # Stage 2d
        # x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2, output_size=x2_size)
        x = self.upsample4(x)
        #x = F.max_unpool2d(x, id2, kernel_size=2, stride=2, output_size=x2_size)
        # x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x = F.relu(self.bn22d(self.conv22d(x)))
        x = F.relu(self.bn21d(self.conv21d(x)))

        # Stage 1d
        #x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2, output_size=x1_size)
        x = self.upsample5(x)
        x = F.relu(self.bn12d(self.conv12d(x)))
        x = self.conv11d(x)

        return x

def parse_args():
    parser = ArgumentParser(description='Efficient semantic segmentation')
    parser.add_argument('--model', default="SegNet", help="model name: (default SegNet)")
    parser.add_argument('--dataset', default="camvid", help="dataset: cityscapes or camvid")
    parser.add_argument('--checkpoint', type=str, default="./checkpoint/camvid/SegNetbs8gpu1_train/model_200.pth",
                        help="use the file to load the checkpoint for testing")
    parser.add_argument('--cuda', action='store_true', help="run on CPU or GPU")
    parser.add_argument('--gpus', default="0", type=str, help="gpu ids (default: 0)")
    parser.add_argument('--image_path', type=str, default='./dataset/camvid/test/Seq05VD_f04620.png', help="path to the test image")
    parser.add_argument('--output_path', type=str, default='./output/result_Seq05VD_f04620.png', help="path to the output image")
    parser.add_argument('--num_workers', type=int, default=1, help="the number of parallel threads")
    parser.add_argument('--num_classes', type=int, default=11,
                        help="number of classes (default: 11 for CamVid dataset)")
    args = parser.parse_args()
    return args


def test_image(args, image, model):
    """Test the model on a single image."""
    model.eval()

    # Load and preprocess the image
    print("=====> loading image: '{}'".format(args.image_path))
    # image = load_image(args.image_path)
    if args.cuda:
        image = image.cuda()

    with torch.no_grad():
        output = model(image)
        output = output.cpu().data[0].numpy()
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

    return output


def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    normalized_image = 255 * (image - image_min) / (image_max - image_min)
    return normalized_image.astype(np.uint8)

def apply_color_map(image, cmap='viridis'):
    colormap = cm.get_cmap(cmap)
    colored_image = colormap(image / 255.0)  # Normalize to [0, 1] for colormap
    colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)  # Drop the alpha channel and scale to [0, 255]
    return colored_image


def apply_color_map_2(output):  # 语义色值映射
    # 获取图像的宽高
    h, w = output.shape
    # 创建一个新的 RGB 图像
    colored_output = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(12):
        colored_output[output == i] = color_map[i]

    return colored_output


def main():
    args = parse_args()

    if args.cuda:
        print("=====> use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("no GPU found or wrong gpu id, please run without --cuda")

    # Build the model
    model = build_model(args.model, num_classes=args.num_classes)
    datas, test_loader = build_dataset_test(args.dataset, args.num_workers)
    for i, (input, label, size, name) in enumerate(test_loader):
        if args.image_path.split('/')[-1] == name[0]+'.png':
            break
        # else:
        #     print(name[0], '!=', args.image_path.split('/')[-1])

    if args.cuda:
        model = model.cuda()  # Using GPU for inference
        cudnn.benchmark = True
    else:
        model = model.cpu()

    # Load the checkpoint
    if os.path.isfile(args.checkpoint):
        print("=====> loading checkpoint '{}'".format(args.checkpoint))
        if not args.cuda:
            checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(args.checkpoint)
        # print(list(checkpoint['model'].keys()))
        model.load_state_dict(checkpoint['model'])
    else:
        print("=====> no checkpoint found at '{}'".format(args.checkpoint))
        raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))

    # halfmodel = HalfModel(model)
    output = test_image(args, input, model)      # 测试原模型输出
    colored_output = apply_color_map_2(output)

    result_image = Image.fromarray(colored_output)              # 色彩映射后的图像
    result_image.save(args.output_path)
    print("Prediction saved to " + args.output_path)


if __name__ == '__main__':
    main()

