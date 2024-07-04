import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py
import os 
from sklearn.model_selection import train_test_split
from PIL import Image
import hdf5storage
import mat73#12:54:30
# import h5py

data_path = 'data/'
if not os.path.exists(data_path):
    print('making dir')
    os.makedirs(data_path)
print('exist dir')
f=h5py.File(data_path + "train01.mat")               #读取mat文件

# f = hdf5storage.loadmat(data_path + "nyu_depth_v2_labeled.mat")
print('opened')
image = np.array(f['images'])
depth = np.array(f['depths'])
print(image.shape)


imgae_train, image_test, depth_train, depth_test = train_test_split(image, depth, test_size=0.2, random_state=42)
print(imgae_train.shape, image_test.shape, depth_train.shape, depth_test.shape)
sio.savemat(data_path + 'train.mat', {'images': imgae_train, 'depths': depth_train})
sio.savemat(data_path + 'test.mat', {'images': image_test, 'depths': depth_test})

# image = np.transpose(image, (0, 3, 2, 1))
# image = Image.fromarray(image[0])
# image.save('./test.png')


