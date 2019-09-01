import numpy as np
import torch
import json
from utils_box.dataset import Dataset_CSV
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import api
from detector import Detector


# TODO: Set your coco_table_file, coco_anno_root and set_name
coco_table_file = 'data/coco_table.json'
coco_anno_root = '/home1/xyt/dataset/coco17/'
set_name = 'val2017'
# ==================================


# Read train.json/coco_table_file and set current GPU (for nms_cuda)
with open(coco_table_file, 'r') as load_f:
    coco_table = json.load(load_f)
with open('train.json', 'r') as load_f:
    cfg = json.load(load_f)
torch.cuda.set_device(cfg['device'][0])


# Prepare the network
net = Detector(pretrained=False)
device_out = 'cuda:%d' % (cfg['device'][0])
net.load_state_dict(torch.load('net.pkl', map_location=device_out))
net = net.cuda(cfg['device'][0])
net.eval()


# Get eval dataset
dataset_eval = Dataset_CSV(cfg['root_eval'], cfg['list_eval'], cfg['name_file'], 
    size=net.view_size, train=False, normalize=True)


# Prepare API structure
inferencer = api.COCOEvaluator(net, dataset_eval, 
                coco_table['val_image_ids'], coco_table['coco_labels'])


# Eval
inferencer.step_epoch()
coco = COCO(os.path.join(coco_anno_root, 'annotations', 'instances_' + set_name + '.json'))
coco_pred = coco.loadRes('coco_bbox_results.json')
coco_eval = COCOeval(coco, coco_pred, 'bbox')
coco_eval.params.imgIds = coco_table['val_image_ids']
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
