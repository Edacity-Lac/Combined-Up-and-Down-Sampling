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
- Image Deraining based on SyntoReal [[paper](https://arxiv.org/pdf/2006.05580)] [[code](https://github.com/rajeevyasarla/Syn2Real)]
- Object Detection based on YOLOv3 [[paper](https://arxiv.org/abs/1804.02767)] [[code](https://github.com/ultralytics/yolov3)]


## Prepare Your Data
- Image Reconstruction 
1. You can access the Mini-Imagenet dataset  from [here](https://lyy.mpi-inf.mpg.de/mtl/download/Lmzjm9tX.html); 
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
1. You can access the NYU V2 Depth dataset from [here](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat);
2. Run `split_data.py` to divide the data into training and evaluation sets.
3. The folds of your dataset need satisfy the following structures: 

```
|-- data
|  |-- test.mat
|  |-- train.mat

```

- Image Segmentation
1. You can download [**camvid**](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) dataset from [here](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid);
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

- Image Deraining
1. You can access the NYU V2 Depth dataset from [here](https://pan.baidu.com/s/1SR7yULy0VZ_JZ4Vawqs7gg#list/path=%2F?qq-pf-to=pcqq.c2c);
2. The folds of your dataset need satisfy the following structures: 

```
   .
    ├── data 
    |   ├── train # Training  
    |   |   ├── derain        
    |   |   |   ├── <dataset_name>   
    |   |   |   |   ├── rain              # rain images 
    |   |   |   |   └── norain            # clean images
    |   |   |   └── dataset_filename.txt
    |   └── test  # Testing
    |   |   ├── derain         
    |   |   |   ├── <dataset_name>          
    |   |   |   |   ├── rain              # rain images 
    |   |   |   |   └── norain            # clean images
    |   |   |   └── dataset_filename.txt

```
- Object Detection
We have a custom-labeled dataset for eye movement detection. If you would like access to this dataset, please contact us at [wong@hust.edu.cn](mailto:wong@hust.edu.cn).

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
- Image Deraining
1. mention the labeled, unlabeled, and validation dataset in lines 119-121 of train.py
     ```
       if category == 'derain':
           num_epochs = 50
           train_data_dir = './data/train/derain/'
           val_data_dir = './data/test/derain/'
       labeled_name = 'real_input1.txt'
       unlabeled_name = 'real_input2.txt'
       val_filename = 'SIRR_test.txt'
    ``` 
2. Run the following command to train the base network without Gaussian processes
    ```
    python train.py  -train_batch_size 2  -category derain -exp_name DDN_SIRR_withoutGP  -lambda_GP 0.00 -epoch_start 0
    ```
- Object Detection
1. Run the following command to train the network:
     ```
   python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5n.yaml  --batch-size 128
     ```

## Result(视版面待定,少的话放个总排名）
