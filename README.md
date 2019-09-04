# FCOS in Pytorch

An unofficial implementation of FCOS in pytorch. 
FCOS: Fully Convolutional One-Stage Object Detection.

https://arxiv.org/abs/1904.01355



## 1. COCO (1x)

First, configure train.json file, add your root. 

```json
{
    "root_train": "/home1/xyt/dataset/coco17/images",
    "root_eval": "/home1/xyt/dataset/coco17/images",
    "list_train": "data/coco_train2017.txt",
    "list_eval": "data/coco_val2017.txt",
    "name_file": "data/coco_name.txt",

    "load": false,
    "save": true,
    "pretrain": true,
    "freeze_bn": true,
    "freeze_stages": 1,
    "epoches": 12,

    "nbatch_train": 16,
    "nbatch_eval": 16,
    "device": [1,2,3,5,6,7,8,9],
    "num_workers": 16,
    
    "lr_base": 0.01,
    "lr_gamma": 0.1,
    "lr_schedule": [60000, 80000],
    "momentum": 0.9,
    "weight_decay": 0.0001,

    "boxarea_th": 32,
    "grad_clip": 3,
    
    "img_scale_min": 0.8,
    "augmentation": false
}
```

Then, configure some parameters in *detector.py* file.

```python
self.view_size = 1025
self.classes = 80   # TODO: total 80 classes exclude background
```

Run analyze to get mAP curves.

```python
map_mean
[0.0938 0.1693 0.2172 0.2306 0.2406 0.2413 0.2667 0.2587 0.3238 0.328
 0.3325 0.3338]
map_50
[0.173  0.3071 0.3645 0.3802 0.3982 0.4032 0.4265 0.4241 0.4998 0.5053
 0.5108 0.5126]
map_75
[0.0921 0.1708 0.2242 0.2393 0.2537 0.2519 0.2828 0.2723 0.3409 0.3455
 0.3502 0.3527]
```

Run cocoeval and got mAP: **33.6%**

```python
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.336
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.516
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.355
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.180
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.371
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.443
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.280
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.429
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.450
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.253
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.498
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.575
```

