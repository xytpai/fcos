# FCOS in Pytorch

An unofficial implementation of FCOS in pytorch. 
FCOS: Fully Convolutional One-Stage Object Detection.

https://arxiv.org/abs/1904.01355

| paper (800px) | ours (nearly 700px) |
| :--: | :--: |
| 36.6 | **35.9** |

![](images/pred_demo.bmp)



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
[0.0956 0.1589 0.2083 0.2308 0.2433 0.2537 0.2599 0.2621 0.3241 0.3315
 0.3376 0.3378]
map_50
[0.1846 0.3076 0.3512 0.3832 0.3959 0.4021 0.4182 0.4164 0.4978 0.5068
 0.5135 0.5142]
map_75
[0.0894 0.1555 0.2187 0.2415 0.2569 0.2695 0.2793 0.2803 0.3443 0.3545
 0.3602 0.3606]
```

Run cocoeval and got mAP: **34.0%**

```python
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.340
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.518
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.362
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.182
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.377
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.456
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.282
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.438
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.465
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.258
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.517
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.593
```



## 2. COCO (2x)

Like 2 in *train.json* modify key

```json
"epoches": 24,
"lr_schedule": [120000, 160000],
```

Run train to get results. It takes about 28 hours with 8x Titan-Xp. Run analyze to get mAP curves.

```python
map_mean
[0.102  0.1695 0.2081 0.2324 0.2392 0.2428 0.2513 0.2599 0.2659 0.263
 0.2705 0.2787 0.2763 0.2753 0.2883 0.2942 0.3457 0.3487 0.3494 0.3502
 0.353  0.3547 0.3552 0.3547]
map_50
[0.1956 0.3047 0.3625 0.3881 0.3935 0.4019 0.4087 0.4301 0.4218 0.4165
 0.4315 0.4455 0.4466 0.4397 0.4552 0.459  0.5234 0.5266 0.5275 0.5294
 0.5316 0.5326 0.5346 0.5337]
map_75
[0.0965 0.1715 0.2123 0.2396 0.2498 0.2516 0.2627 0.2707 0.2789 0.2798
 0.2866 0.2941 0.2957 0.292  0.3046 0.3114 0.3696 0.3713 0.3743 0.3735
 0.3779 0.3801 0.3805 0.3794]
```

Run cocoeval and got mAP: **35.9%**

```python
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.359
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.537
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.385
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.194
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.399
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.474
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.296
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.457
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.483
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.278
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.535
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.611
```

