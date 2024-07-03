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
- Image Reconstruction based on Standard encoder-decoder architecture [[paper](https://arxiv.org/pdf/1908.09895v2.pdf)]
- Monocular Depth Estimation based on FastDepth [[paper](https://arxiv.org/pdf/1903.03273)] [[code](https://github.com/dwofk/fast-depth)]
- Image Segmentation based on SegNet [[paper](https://arxiv.org/pdf/1511.00561)] [[code](https://github.com/xiaoyufenfei/Efficient-Segmentation-Networks)]


## Prepare Your Data
- Image Reconstruction 
1. You can access the Mini-Imagenet dataset from [[here](https://lyy.mpi-inf.mpg.de/mtl/download/Lmzjm9tX.html)] ;
2. The folds of your dataset need satisfy the following structures: 

````
$PATH_TO_DATASET/mini-imagenet
├──── train
│    ├──── n01532829
│    ├──── n01558993
│    ├──── ...
├──── test
├──── val
````
- Monocular Depth Estimation 
1. You can access the NYU V2 Depth dataset from [[here](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat)] ;
2. Run `split_data.py` to divide the data into training and evaluation sets.

- Image Segmentation
1. You can download [**camvid**](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) dataset from [here](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid).
2. The folds of your dataset need satisfy the following structures: 

```
|-- dataset
|  |-- camvid
|  |  |-- train
|  |  |-- trainannot
|  |  |-- val
|  |  |-- valannot
|  |  |-- test
|  |  |-- testannot
|  |  |-- ...

```



## Training
- Image Reconstruction
1. Run the following command to train the network:
     ```
    python train.py
     ```
  
- Monocular Depth Estimation
1. Run the following command to train the network:
     ```
    python main.py -mode train -backbone mobilenetv2 --criterion l1 --gpu True -e 30 --encoder_decoder_choice <your choice>  --bsize 16
     ```
- Image Segmentation
    
1. Run the following command to train the network:
     ```
conda activate segnet && python train.py --model SegNet --dataset camvid --input_size '360,480' --num_workers 4 --classes 11 --lr 5e-4 --batch_size 8 --train_type train --max_epochs 200 --cuda True

     ```




## Result(视版面待定,少的话放个总排名）
