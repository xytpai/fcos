import numpy as np
import torch
import json
import api
from utils_box.dataset import Dataset_CSV
from detector import Detector


# Read train.json and set current GPU (for nms_cuda)
with open('train.json', 'r') as load_f:
    cfg = json.load(load_f)
torch.cuda.set_device(cfg['device'][0])
net = Detector(pretrained=False)


# TODO: You can change nms_th, device or nbatch_eval
# net.nms_th = 0.05
# cfg['device'] = [0,1,2,3,9]
# cfg['nbatch_eval'] = 30
# ==================================


# Prepare the network
device_out = 'cuda:%d' % (cfg['device'][0])
net.load_state_dict(torch.load('net.pkl', map_location=device_out))
net = torch.nn.DataParallel(net, device_ids=cfg['device'])
net = net.cuda(cfg['device'][0])
net.eval()


# Get eval dataset and dataloader
dataset_eval = Dataset_CSV(cfg['root_eval'], cfg['list_eval'], cfg['name_file'], 
    size=net.module.view_size, train=False, normalize=True)
loader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=cfg['nbatch_eval'], 
                    shuffle=False, num_workers=0, collate_fn=dataset_eval.collate_fn)


# Prepare API structure
evaluator = api.Evaluator(net, dataset_eval, loader_eval, cfg['device'])


# Eval
evaluator.step_epoch()
