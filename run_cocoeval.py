import numpy as np
import torch
import json
import torchvision.transforms as transforms
from utils_box.dataset import Dataset_CSV
from utils_box.eval_csv import eval_detection
from encoder import Encoder
from detector import Detector
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


coco_table_file = 'data/coco_table.json'
coco_anno_root = '/home1/xyt/dataset/coco17/'
set_name = 'val2017'


with open(coco_table_file, 'r') as load_f:
    coco_table = json.load(load_f)

with open('train.json', 'r') as load_f:
    cfg = json.load(load_f)
net = Detector(pretrained=False)


device_out = 'cuda:%d' % (cfg['device'][0])
net.load_state_dict(torch.load('net.pkl', map_location=device_out))
net = torch.nn.DataParallel(net, device_ids=cfg['device'])
net = net.cuda(cfg['device'][0])
net.eval()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])
dataset_eval = Dataset_CSV(cfg['root_eval'], cfg['list_eval'], cfg['name_file'], 
    size=net.module.eval_size, train=False, transform=transform)


encoder = Encoder(
    net.module.regions,
    net.module.first_stride,
    net.module.train_size, 
    net.module.eval_size,
    net.module.nms, 
    net.module.nms_th, 
    net.module.nms_iou,
    net.module.max_detections)


# Eval
with torch.no_grad():

    results = []

    for i in range(len(dataset_eval)):

        img, bbox, label, scale = dataset_eval[i]
        img = img.cuda(cfg['device'][0])
        img = img.view(1, img.shape[0], img.shape[1], img.shape[2])
        cls_out, reg_out = net(img)
        cls_i_preds, cls_p_preds, reg_preds = encoder.decode(cls_out.cpu(), reg_out.cpu())

        cls_i_preds = cls_i_preds[0]
        cls_p_preds = cls_p_preds[0]
        reg_preds = reg_preds[0]

        if reg_preds.shape[0] > 0:

            ymin, xmin, ymax, xmax = reg_preds.split([1, 1, 1, 1], dim=1)
            h = ymax - ymin
            w = xmax - xmin 
            reg_preds = torch.cat([xmin, ymin, w, h], dim=1)
            reg_preds = reg_preds / float(scale)

            for box_id in range(reg_preds.shape[0]):

                score = float(cls_p_preds[box_id])
                label = int(cls_i_preds[box_id])
                box = reg_preds[box_id, :]

                image_result = {
                    'image_id'    : coco_table['val_image_ids'][i],
                    'category_id' : coco_table['coco_labels'][str(label)],
                    'score'       : float(score),
                    'bbox'        : box.tolist(),
                }

                results.append(image_result)

        print('step:%d/%d' % (i, len(dataset_eval))

    json.dump(results, open('coco_bbox_results.json', 'w'), indent=4)

    coco = COCO(os.path.join(coco_anno_root, 'annotations', 'instances_' + set_name + '.json'))
    coco_pred = COCOeval.loadRes('coco_bbox_results.json')
    
    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
