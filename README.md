# BP-Net

## Introduction

This is the pytorch implementation of our paper [Bilateral Propagation Network for Depth Completion](https://arxiv.org/abs/2403.11270).

## Environment


You can directly build the environment by running the following command if you use *conda* as the environment management
tool.
```
conda env create -f environment.yml
```
Then you should have an env named *bp*.

## Setup
Compile the C++ and CUDA code:
```
cd exts
python setup.py install
```

## Dataset
We train and evaluate on KITTI and NYUV2 dataset.

### KITTI
Please download KITTI [depth completion](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion)
dataset.
The structure of data directory:
```
└── datas
    └── kitti
        ├── data_depth_annotated
        │   ├── train
        │   └── val
        ├── data_depth_velodyne
        │   ├── train
        │   └── val
        ├── raw
        │   ├── 2011_09_26
        │   ├── 2011_09_28
        │   ├── 2011_09_29
        │   ├── 2011_09_30
        │   └── 2011_10_03
        ├── test_depth_completion_anonymous
        │   ├── image
        │   ├── intrinsics
        │   └── velodyne_raw
        └── val_selection_cropped
            ├── groundtruth_depth
            ├── image
            ├── intrinsics
            └── velodyne_raw
```

### NYUV2
We used preprocessed NYUv2 [HDF5 dataset](http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz) provided by [Fangchang Ma](https://github.com/fangchangma/sparse-to-dense).
Note, the original full NYUv2 dataset is available at the [official website](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).

The structure of data directory:

```
└── datas
    └── nyudepthv2
        ├── train
        │    ├── basement_0001a
        │    │    ├── 00001.h5
        │    │    └── ...
        │    ├── basement_0001b
        │    │    ├── 00001.h5
        │    │    └── ...
        │    └── ...
        └── val
            └── official
                ├── 00001.h5
                └── ...
```



## Trained Models
You can directly download the trained models and extract them in *checkpoints* directory.
Our models are trained on a GPU workstation with 4 Nvidia GTX 3090.
- [BP_KITTI](https://drive.google.com/file/d/1sNq68jEEyLU2_XmH1UZRs4wOrt_MidcO/view)
- [BP_NYU](https://drive.google.com/file/d/1XfKoJVCkj_J5euTkONI_iBx9ZxLUov9w/view)

## Train
You can also train by yourself.

### KITTI

train on KITTI
```
torchrun --nproc_per_node=4 --master_port 4321 train.py \
gpus=[0,1,2,3] num_workers=4 name=BP_KITTI \
net=PMP data=KITTI \
lr=1e-3 train_batch_size=2 test_batch_size=2 \
sched/lr=NoiseOneCycleCosMo sched.lr.policy.max_momentum=0.90 \
nepoch=30 test_epoch=25 ++net.sbn=true
```


### NYUV2

train on NYUV2
```
torchrun --nproc_per_node=2 --master_port 1100 train.py \
gpus=[0,1] num_workers=4 name=BP_NYU \
net=PMP data=NYU data.num_mask=1 \
lr=2e-3 train_batch_size=8 test_batch_size=1 \
nepoch=100 test_epoch=80 ++net.sbn=true
```

## Test

With the trained model, 
you can test and save results.

### KITTI

test on KITTI selval set
```
python test.py gpus=[0] name=BP_KITTI ++chpt=BP_KITTI \
net=PMP num_workers=4 \
data=KITTI data.testset.mode=selval \
test_batch_size=1 metric=RMSE ++net.compile=true
```


test on KITTI test set and save for submission
```
python test.py gpus=[0] name=BP_KITTI ++chpt=BP_KITTI \
net=PMP num_workers=4 \
data=KITTI data.testset.mode=test data.testset.height=352 \
test_batch_size=1 metric=RMSE ++save=true
```

### NYUV2

test on NYUV2 test set

```
python test.py gpus=[0] name=BP_NYU ++chpt=BP_NYU \
net=PMP num_workers=4 \
data=NYU test_batch_size=1 metric=MetricALL
```

## Citation
If you find this work useful in your research, please consider citing:
```
@article{BP-Net,
title={Bilateral Propagation Network for Depth Completion},
author={Tang, Jie and Tian, Fei-Peng and An, Boshi and Li, Jian and Tan, Ping},
journal={arXiv preprint arXiv:2403.11270},
year={2024}
}
```