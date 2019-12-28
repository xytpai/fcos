# FCOS in Pytorch

An unofficial implementation of FCOS in Pytorch. 
FCOS: Fully Convolutional One-Stage Object Detection.

https://arxiv.org/abs/1904.01355

**Some modifies are adopted:**

- Remove *center-ness* branch for simplicity.
- Add center sample mechanism to improve performance.
- Predect yxhw instead of yxyx +0.3Map.
- Note: GPUCompute Capability >= 6.1

**Download:**  https://pan.baidu.com/s/1MjsgE2fu86qmWZpVD7a8Aw

| paper (800px) | official (800px) | ours (nearly 700px) |
| :-----------: | :--------------: | :-----------------: |
|     36.6      |       38.7       |      **37.3**       |

![](images/pred_street.jpg)

Some *cuda* operations are provided for acceleration. 

```bash
cd libs/nms_cuda
python setup.py install
cd libs/sigmoid_focal_loss_cuda
python setup.py install
cd libs/assign_box_cuda
python setup.py install
```

## COCO (2x)

1. Copy configs/detector_coco.py to ./detector.py
2. Copy configs/train_coco_2x.json to ./train.json
3. Configure train.json file, add your root. 
4. Command: *python run_train.py*  to start schedule, it takes about 30 hours with 8x Titan-XP.
5. Command: *python run_analyze.py*  to get mAP curves.
6. Command: *python run_cocoeval.py*  to get mAP.

```python
map_mean
[0.1345 0.1984 0.2192 0.2468 0.2496 0.2606 0.2729 0.2708 0.2768 0.2872
 0.2924 0.295  0.2973 0.3041 0.2927 0.3027 0.3565 0.3583 0.3622 0.3642
 0.3659 0.3679 0.3689 0.37  ]
map_50
[0.2482 0.3404 0.3749 0.3997 0.4059 0.4182 0.4401 0.4393 0.4389 0.4496
 0.4584 0.4646 0.4618 0.4719 0.4584 0.4741 0.535  0.5385 0.5405 0.5436
 0.5457 0.5489 0.5497 0.5511]
map_75
[0.1304 0.2044 0.2265 0.2596 0.2595 0.2758 0.2895 0.283  0.2914 0.3037
 0.311  0.315  0.3145 0.3246 0.3136 0.3232 0.3804 0.3829 0.3869 0.3906
 0.3917 0.3924 0.3943 0.3962]
```

```python
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.373
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.556
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.398
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.198
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.411
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.486
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.303
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.482
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.525
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.326
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.575
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.639
```

