import numpy as np
import torch
import json
import torchvision.transforms as transforms
from utils_box.dataset import Dataset_CSV
from utils_box.eval_csv import eval_detection
from encoder import Encoder
from detector import Detector, get_loss


with open('train.json', 'r') as load_f:
    cfg = json.load(load_f)


net = Detector(pretrained=cfg['pretrain'])
log = []
device_out = 'cuda:%d' % (cfg['device'][0])
if cfg['load']:
    net.load_state_dict(torch.load('net.pkl', map_location=device_out))
    log = list(np.load('log.npy'))
net = torch.nn.DataParallel(net, device_ids=cfg['device'])
net = net.cuda(cfg['device'][0])
net.train()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])
dataset_train = Dataset_CSV(cfg['root_train'], cfg['list_train'], cfg['name_file'], 
    size=net.module.train_size, train=True, transform=transform, 
    boxarea_th = cfg['boxarea_th'], 
    img_scale_min = cfg['img_scale_min'], 
    crop_scale_min = cfg['crop_scale_min'], 
    aspect_ratio = cfg['aspect_ratio'], 
    remain_min = cfg['remain_min'])
dataset_eval = Dataset_CSV(cfg['root_eval'], cfg['list_eval'], cfg['name_file'], 
    size=net.module.eval_size, train=False, transform=transform)
loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=cfg['nbatch_train'], 
                    shuffle=True, num_workers=0, collate_fn=dataset_train.collate_fn)
loader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=cfg['nbatch_eval'], 
                    shuffle=False, num_workers=0, collate_fn=dataset_eval.collate_fn)


lr = cfg['lr']
lr_decay = cfg['lr_decay']
encoder = Encoder(net.module.a_hw, net.module.scales, net.module.first_stride, 
            train_iou_th=net.module.iou_th, 
            train_size=net.module.train_size,
            eval_size=net.module.eval_size,
            nms=net.module.nms, nms_th=net.module.nms_th, nms_iou=net.module.nms_iou,
            max_detections=net.module.max_detections)


epoch = 0
for epoch_num in cfg['epoch_num']:

    for e in range(epoch_num):

        opt = torch.optim.SGD(net.parameters(), lr=lr, 
            momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
        if cfg['freeze_bn']:
            net.module.backbone.freeze_bn()

        # Train
        for i, (img, bbox, label, scale) in enumerate(loader_train):
            opt.zero_grad()
            cls_targets, reg_targets = encoder.encode(label, bbox)
            temp = net(img, cls_targets, reg_targets)
            loss = get_loss(temp)
            loss.backward()
            clip = cfg['grad_clip']
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            print('epoch:%d, step:%d/%d, loss:%f' % \
                (epoch, i*cfg['nbatch_train'], len(dataset_train), loss))
        
        # Eval
        with torch.no_grad():
            net.eval()
            pred_bboxes = []
            pred_labels = []
            pred_scores = []
            gt_bboxes = []
            gt_labels = []
            for i, (img, bbox, label, scale) in enumerate(loader_eval):
                cls_out, reg_out = net(img)
                cls_i_preds, cls_p_preds, reg_preds = encoder.decode(cls_out.cpu(), reg_out.cpu())
                for idx in range(len(cls_i_preds)):
                    cls_i_preds[idx] = cls_i_preds[idx].detach().numpy()
                    cls_p_preds[idx] = cls_p_preds[idx].detach().numpy()
                    reg_preds[idx] = reg_preds[idx].detach().numpy()
                bbox = list(bbox)
                label = list(label)
                for idx in range(len(bbox)):
                    bbox[idx] = bbox[idx].detach().numpy()
                    label[idx] = label[idx].detach().numpy()
                pred_bboxes += reg_preds
                pred_labels += cls_i_preds
                pred_scores += cls_p_preds
                gt_bboxes += bbox
                gt_labels += label
                print('  Eval: {}/{}'.format(i*cfg['nbatch_eval'], len(dataset_eval)), end='\r')
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
            log.append([map_mean, map_50, map_75])
            net.train()
        
        # Save
        if cfg['save']:
            torch.save(net.module.state_dict(),'net.pkl')
            if len(log)>0:
                np.save('log.npy', log)
         
        epoch += 1
    lr *= lr_decay
