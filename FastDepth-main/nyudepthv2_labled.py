import os
import os.path
import random
import scipy.io as sio
import numpy as np
import h5py
from PIL import Image

import torch
from torchvision import transforms 
import torchvision.transforms.functional
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, random_split

iheight, iwidth = 480, 640

def worker_init_fn(work_id):
    np.random.seed(work_id)

def h5_loader(path):
    mat = sio.loadmat(path)
    rgb = np.array(mat['images'])
    rgb = np.transpose(rgb, (0, 3, 2, 1))
    depth = np.array(mat['depths'])
    depth = np.transpose(depth, (0, 2, 1))

    return rgb, depth

class NYUDataset(Dataset):
    def __init__(self, root_dir, train, loader=h5_loader):
        if train:
            path = os.path.join(root_dir, 'train.mat')
            self.transform = self.train_transform
        else:
            path = os.path.join(root_dir, 'test.mat')
            self.transform = self.val_transform
        self.images, self.depths = loader(path)
        self.output_size = (224, 224)

    def __len__(self):
         return self.images.shape[0]


    def __getitem__(self, index):
        rgb = self.images[index]
        depth = self.depths[index]
        rgb_tensor, depth_tensor = self.transform(rgb, depth)
        return rgb_tensor, depth_tensor


    def train_transform(self, rgb, depth):
        s = np.random.uniform(1.0, 1.5) # random scaling
        depth_np = depth / s
        #converting to PIL.
        rgb_pil = Image.fromarray(rgb.copy())
        depth_pil = Image.fromarray(depth_np.copy())
        #resize1
        dim1 = (int(250*480/iheight),int(250*640/iheight))
        resize1 = transforms.Resize(dim1)
        rgb_pil = resize1(rgb_pil)
        depth_pil = resize1(depth_pil)

        # Random rotation
        angle = transforms.RandomRotation.get_params((-5,5))
        rgb_pil = transforms.functional.rotate(rgb_pil,angle)
        depth_pil = transforms.functional.rotate(depth_pil,angle)

        #resize2
        dim2 = (int(s*250*480/iheight),int(s*250*640/iheight))
        resize2 = transforms.Resize(dim2)
        rgb_pil = resize2(rgb_pil)
        depth_pil = resize2(depth_pil)
        # Center Crop
        center_crop = transforms.CenterCrop((228,304))
        rgb_pil = center_crop(rgb_pil)
        depth_pil = center_crop(depth_pil)

        # Random horizonal flip
        if random.random() > 0.5:
            rgb_pil = transforms.functional.hflip(rgb_pil)
            depth_pil = transforms.functional.hflip(depth_pil)

        #resize3

        resize3 = transforms.Resize(self.output_size)
        rgb_pil = resize3(rgb_pil)
        depth_pil = resize3(depth_pil)

        # Color Jitter
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)
        rgb_pil = color_jitter(rgb_pil)
        
        # To Tensor
        to_tensor = transforms.ToTensor()
        rgb_tensor = to_tensor(np.array(rgb_pil))
        depth_tensor = to_tensor(np.array(depth_pil))
        return rgb_tensor, depth_tensor

    def val_transform(self,rgb, depth):
        #converting to PIL.

        rgb_pil = Image.fromarray(rgb)
        depth_pil = Image.fromarray(depth)

        #resize1
        dim1 = (int(250*480/iheight),int(250*640/iheight))
        resize1 = transforms.Resize(dim1)
        rgb_pil = resize1(rgb_pil)
        depth_pil = resize1(depth_pil)

        # Center Crop
        center_crop = transforms.CenterCrop((228,304))
        rgb_pil = center_crop(rgb_pil)
        depth_pil = center_crop(depth_pil)

        #resize2
        resize2 = transforms.Resize(self.output_size)
        rgb_pil = resize2(rgb_pil)
        depth_pil = resize2(depth_pil)

        # To Tensor
        
        to_tensor = transforms.ToTensor()
        rgb_tensor = to_tensor(np.array(rgb_pil))
        depth_tensor = to_tensor(np.array(depth_pil))

        return rgb_tensor, depth_tensor

def create_data_loaders(args):
    print('Creating dataset... patience.')
    home_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    data_path = os.path.join(home_path, 'data')

    if not os.path.isdir(data_path):
        raise RuntimeError('Dataset directory not found.')
    train_loader, val_loader = None, None
    train_dataset = NYUDataset(data_path, train=True)
    if (args.mode == 'train'):
      if args.samples != None:
         train_split, _ = random_split(train_dataset,(args.samples,len(train_dataset)-args.samples))
         args.train_set = train_split
         train_loader = torch.utils.data.DataLoader(train_split, batch_size= args.bsize, shuffle=True,num_workers=args.workers, pin_memory=True, sampler=None,worker_init_fn=worker_init_fn)
      else:
          args.train_set = train_dataset
          train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= args.bsize, shuffle=True,num_workers=args.workers, pin_memory=True, sampler=None,worker_init_fn=worker_init_fn)
    val_dataset = NYUDataset(data_path, train=False)
    args.val_set = val_dataset
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    print('Finish loading datasets')
    return train_loader, val_loader


