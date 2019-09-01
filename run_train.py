import numpy as np
import torch
import json
import time
import api
from utils_box.dataset import Dataset_CSV
from detector import Detector
import utils_box.augment as augment
import random


# Read train.json and set current GPU (for nms_cuda)
with open('train.json', 'r') as load_f:
    cfg = json.load(load_f)
torch.cuda.set_device(cfg['device'][0])


# Prepare the network and read log
net = Detector(pretrained=cfg['pretrain'])
log = []
device_out = 'cuda:%d' % (cfg['device'][0])
if cfg['load']:
    net.load_state_dict(torch.load('net.pkl', map_location=device_out))
    log = list(np.load('log.npy'))
net = torch.nn.DataParallel(net, device_ids=cfg['device'])
net = net.cuda(cfg['device'][0])
net.train()


# TODO: Define augment
def aug_func(img, boxes):
    if random.random() < 0.3:
        img, boxes = augment.colorJitter(img, boxes, 
                        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
    if random.random() < 0.5:
        img, boxes = augment.random_rotation(img, boxes, degree=5)
    if random.random() < 0.5:
        img, boxes = augment.random_crop_resize(img, boxes, size=512, 
                        crop_scale_min=0.2, aspect_ratio=[3./4, 4./3], remain_min=0.7, 
                        attempt_max=10)
    return img, boxes
if cfg['augmentation']:
    augmentation = aug_func
else:
    augmentation = None


# Get train/eval dataset and dataloader
dataset_train = Dataset_CSV(cfg['root_train'], cfg['list_train'], cfg['name_file'], 
    size=net.module.view_size, train=True, normalize=True, 
    boxarea_th = cfg['boxarea_th'], 
    img_scale_min = cfg['img_scale_min'], augmentation=augmentation)
dataset_eval = Dataset_CSV(cfg['root_eval'], cfg['list_eval'], cfg['name_file'], 
    size=net.module.view_size, train=False, normalize=True)
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=cfg['nbatch_train'], 
                    shuffle=True, num_workers=cfg['num_workers'], collate_fn=dataset_train.collate_fn)
loader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=cfg['nbatch_eval'], 
                    shuffle=False, num_workers=0, collate_fn=dataset_eval.collate_fn)


# Prepare optimizer
lr_base = cfg['lr_base']
lr_gamma = cfg['lr_gamma']
lr_schedule = cfg['lr_schedule']
opt = torch.optim.SGD(net.parameters(), lr=lr_base, 
            momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])


# Prepare lr_func
WARM_UP_ITERS = 500
WARM_UP_FACTOR = 1.0 / 3.0
def lr_func(step):
    lr = lr_base
    if step < WARM_UP_ITERS:
        alpha = float(step) / WARM_UP_ITERS
        warmup_factor = WARM_UP_FACTOR * (1.0 - alpha) + alpha
        lr = lr*warmup_factor 
    else:
        for i in range(len(lr_schedule)):
            if step < lr_schedule[i]:
                break
            lr *= lr_gamma
    return float(lr)


# Prepare API structure
trainer = api.Trainer(net, dataset_train, loader_train, cfg['device'], opt, cfg['grad_clip'], lr_func)
evaluator = api.Evaluator(net, dataset_eval, loader_eval, cfg['device'])
if cfg['load']:
    trainer.step = log[-1][3]
    trainer.epoch = log[-1][4]


# Run epoch
while True:
    if trainer.epoch >= cfg['epoches']:
        break
    net.module.backbone.freeze_stages(int(cfg['freeze_stages']))
    if cfg['freeze_bn']:
        net.module.backbone.freeze_bn()
    trainer.step_epoch()
    map_mean, map_50, map_75 = evaluator.step_epoch()
    log.append([map_mean, map_50, map_75, trainer.step, trainer.epoch])
    if cfg['save']:
        torch.save(net.module.state_dict(),'net.pkl')
        np.save('log.npy', log)
print('Schedule finished!')
