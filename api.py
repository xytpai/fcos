import torch
import numpy as np 
import time
import json
import torchvision.transforms as transforms
from utils_box.eval_csv import eval_detection
from utils_box.dataset import center_fix
from detector import get_loss, get_pred
'''
- Note: If you change detector.py or bacbone.py, 
        you may need to change this file.
'''



class Trainer(object):
    def __init__(self, net, dataset, loader, device,   
                    opt, grad_clip=3, lr_func=None):
        '''
        external initialization structure: 
            net(DataParallel), dataset(Dataset), loader(DataLoader), device(List), opt(Optimizer)
        grad_clip: limit the gradient size of each iteration
        lr_func: lr_func(step) -> float
        self.step, self.epoch for outside use
        '''
        self.net = net
        self.dataset = dataset
        self.loader = loader
        self.device = device
        self.opt = opt
        self.grad_clip = grad_clip
        self.lr_func = lr_func
        self.step = 0
        self.epoch = 0
    

    def step_epoch(self):
        '''
        train one epoch
        '''
        lr = -1
        for i, (imgs, boxes, labels, locs, scales) in enumerate(self.loader):
            if self.lr_func is not None:
                lr = self.lr_func(self.step)
                for param_group in self.opt.param_groups:
                    param_group['lr'] = lr
            if i == 0:
                batch_size = int(imgs.shape[0])
            time_start = time.time()
            self.opt.zero_grad()
            temp = self.net(imgs, locs, labels, boxes)
            loss = get_loss(temp)
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
            self.opt.step()
            maxmem = int(torch.cuda.max_memory_allocated(device=self.device[0]) / 1024 / 1024)
            time_end = time.time()
            totaltime = int((time_end - time_start) * 1000)
            print('total_step:%d: epoch:%d, step:%d/%d, loss:%f, maxMem:%dMB, time:%dms, lr:%f' % \
                (self.step, self.epoch, i*batch_size, len(self.dataset), loss, maxmem, totaltime, lr))
            self.step += 1
        self.epoch += 1



class Evaluator(object):
    def __init__(self, net, dataset, loader, device):
        '''
        external initialization structure: 
            net(DataParallel), dataset(Dataset), loader(DataLoader), device(List),
        '''
        self.net = net
        self.dataset = dataset
        self.loader = loader
        self.device = device
    

    def step_epoch(self):
        '''
        return map_mean, map_50, map_75
        note: this function will set self.net.train() at last
        '''
        with torch.no_grad():
            self.net.eval()
            pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels = [], [], [], [], []
            for i, (imgs, boxes, labels, locs, scales) in enumerate(self.loader):
                if i == 0:
                    batch_size = int(imgs.shape[0])
                temp = self.net(imgs, locs)
                pred_cls_i, pred_cls_p, pred_reg = get_pred(temp, 
                        self.net.module.nms_th, self.net.module.nms_iou) # DataParallel
                for idx in range(len(pred_cls_i)):
                    pred_cls_i[idx] = pred_cls_i[idx].cpu().detach().numpy()
                    pred_cls_p[idx] = pred_cls_p[idx].cpu().detach().numpy()
                    pred_reg[idx] = pred_reg[idx].cpu().detach().numpy()
                _boxes, _label = [], []
                for idx in range(boxes.shape[0]):
                    mask = labels[idx] > 0
                    _boxes.append(boxes[idx][mask].detach().numpy())
                    _label.append(labels[idx][mask].detach().numpy())
                pred_bboxes += pred_reg
                pred_labels += pred_cls_i
                pred_scores += pred_cls_p
                gt_bboxes += _boxes
                gt_labels += _label
                print('  Eval: {}/{}'.format(i*batch_size, len(self.dataset)), end='\r')
            ap_iou = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            ap_res = []
            for iou_th in ap_iou:
                res = eval_detection(pred_bboxes, pred_labels, 
                            pred_scores, gt_bboxes, gt_labels, iou_th=iou_th)
                ap_res.append(res)
            ap_sum = 0.0
            for i in range(len(ap_res)):
                ap_sum += float(ap_res[i]['map'])
            map_mean = ap_sum / float(len(ap_res))
            map_50 = float(ap_res[0]['map'])
            map_75 = float(ap_res[5]['map'])
            print('map_mean:', map_mean, 'map_50:', map_50, 'map_75:', map_75)
            self.net.train()
            return map_mean, map_50, map_75



class COCOEvaluator(object):
    def __init__(self, net, dataset,
                    val_image_ids, coco_labels):
        '''
        external initialization structure: 
            net(Model), dataset(Dataset)
        val_image_ids: [12341, 244135, ...]
        coco_labels: {"1": 1, "2": 4, ...}
        '''
        self.net = net
        self.dataset = dataset
        self.val_image_ids = val_image_ids
        self.coco_labels = coco_labels


    def step_epoch(self):
        '''
        all models should be in cuda()
        write to coco_bbox_results.json
        '''
        with torch.no_grad():
            results = []
            for i in range(len(self.dataset)):
                img, bbox, label, loc, scale = self.dataset[i]
                img = img.cuda().view(1, img.shape[0], img.shape[1], img.shape[2])
                loc = loc.cuda().view(1, -1)
                temp = self.net(img, loc)
                pred_cls_i, pred_cls_p, pred_reg = get_pred(temp, 
                        self.net.nms_th, self.net.nms_iou)
                pred_cls_i = pred_cls_i[0].cpu()
                pred_cls_p = pred_cls_p[0].cpu()
                pred_reg = pred_reg[0].cpu()
                if pred_reg.shape[0] > 0:
                    ymin, xmin, ymax, xmax = pred_reg.split([1, 1, 1, 1], dim=1)
                    h, w = ymax - ymin, xmax - xmin 
                    pred_reg = torch.cat([xmin - loc[0, 1].cpu(), ymin - loc[0, 0].cpu(), w, h], dim=1)
                    pred_reg = pred_reg / float(scale)
                    for box_id in range(pred_reg.shape[0]):
                        score = float(pred_cls_p[box_id])
                        label = int(pred_cls_i[box_id])
                        box = pred_reg[box_id, :]
                        image_result = {
                            'image_id'    : self.val_image_ids[i],
                            'category_id' : self.coco_labels[str(label)],
                            'score'       : float(score),
                            'bbox'        : box.tolist(),
                        }
                        results.append(image_result)
                print('step:%d/%d' % (i, len(self.dataset)), end='\r')
            json.dump(results, open('coco_bbox_results.json', 'w'), indent=4)
    


class Inferencer(object):
    def __init__(self, net):
        '''
        external initialization structure: 
            net(Model)
        '''
        self.net = net
        self.normalizer = transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))


    def pred(self, img_pil):
        '''
        all models should be in cuda()
        return cls_i_preds, cls_p_preds, reg_preds
        '''
        _boxes = torch.zeros(0, 4)
        img_pil, boxes, loc, scale = center_fix(img_pil, _boxes, self.net.view_size)
        img = transforms.ToTensor()(img_pil)
        img = self.normalizer(img).view(1, img.shape[0], img.shape[1], img.shape[2])
        img = img.cuda()
        loc = loc.view(1, -1).cuda()
        with torch.no_grad():
            temp = self.net(img, loc)
            pred_cls_i, pred_cls_p, pred_reg = get_pred(temp, 
                self.net.nms_th, self.net.nms_iou)
            pred_reg[0][:, 0::2] -= loc[0, 0]
            pred_reg[0][:, 1::2] -= loc[0, 1]
            pred_reg[0] /= scale
        return pred_cls_i[0], pred_cls_p[0], pred_reg[0]
