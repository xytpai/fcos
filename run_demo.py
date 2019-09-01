import torch
import numpy as np 
import os
import json
from PIL import Image
import api
import torchvision.transforms as transforms
from detector import Detector
from utils_box.dataset import show_bbox


# Read train.json and set current GPU (for nms_cuda) and prepare the network
DEVICE = 0 # set device
with open('train.json', 'r') as load_f:
    cfg = json.load(load_f)
torch.cuda.set_device(DEVICE)
net = Detector(pretrained=False)
net.load_state_dict(torch.load('net.pkl', map_location='cpu'))
net = net.cuda()
net.eval()


# TODO: Set nms_th
net.nms_th = 0.5
# ==================================


# Read LABEL_NAMES
with open(cfg['name_file']) as f:
    lines = f.readlines()
LABEL_NAMES = []
for line in lines:
    LABEL_NAMES.append(line.strip())


# Prepare API structure
inferencer = api.Inferencer(net)


# Run
for filename in os.listdir('images/'):
    if filename.endswith('jpg'):
        img = Image.open(os.path.join('images/', filename))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        cls_i_preds, cls_p_preds, reg_preds = inferencer.pred(img)
        name = 'images/pred_'+filename.split('.')[0]+'.bmp'
        show_bbox(img, reg_preds.cpu(), cls_i_preds.cpu(), LABEL_NAMES, name)
