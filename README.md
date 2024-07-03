# Combined-Up-and-Down-Sampling

This is a repository for course project.

We evaluate the performance of some famous upsampling dowmsampling operators on some representative tasks and a new Combined-Up-and-Down-Sampling operators Co-Dysample invented by ourself.



## Upsampling and Downsampling Operator
- Conv-Deconv
- Conv-Bilinear
- MAX-pooling-Max-unpooling
- Space2Depth-Depth2Space  [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Real-Time_Single_Image_CVPR_2016_paper.pdf)].
- CARAFE++ [[paper](https://arxiv.org/pdf/2012.04733.pdf)][[code](https://github.com/open-mmlab/mmdetection)]![GitHub stars](http://img.shields.io/github/stars/open-mmlab/mmdetection.svg?logo=github&label=Stars)]
- IndexNet [[paper](https://arxiv.org/pdf/1908.09895v2.pdf)][[code](https://github.com/poppinace/indexnet_matting)]![GitHub stars](http://img.shields.io/github/stars/poppinace/indexnet_matting.svg?logo=github&label=Stars)


## The Tasks We Evaluate
- Standard encoder-decoder architecture on Image Reconstruction [[paper](https://arxiv.org/pdf/1908.09895v2.pdf)]
  



## Prepare Your Data
- Image Reconstruction 
1. Please download from [[here](https://lyy.mpi-inf.mpg.de/mtl/download/Lmzjm9tX.html)] for the Mini-Imagenet dataset;
2. The final path structure used in my code looks like this:

````
$PATH_TO_DATASET/mini-imagenet
├──── train
│    ├──── n01532829
│    ├──── n01558993
│    ├──── ...
├──── test
├──── val
````


## Training
- Image Reconstruction
Run the following command to train the network:

    python train.py 
    





## Result(视版面待定,少的话放个总排名）
