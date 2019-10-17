import math
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from utils_box.anchors import gen_anchors, distance2bbox
from libs.sigmoid_focal_loss import sigmoid_focal_loss
from libs.iou_loss import iou_loss
from libs.nms import box_nms 
# TODO: choose backbone
from backbone import resnet50 as Backbone



class Detector(nn.Module):
    def __init__(self, pretrained=False):
        super(Detector, self).__init__()

        # ---------------------------
        # TODO: Param
        self.regions = [0, 64, 128, 256, 512, 9999]
        self.first_stride = 8
        self.view_size = 1025
        self.classes = 80
        self.nms_th = 0.05
        self.nms_iou = 0.6
        self.max_detections = 3000
        self.center_offset_ratio = 1.5
        # ---------------------------

        # fpn =======================================================
        self.backbone = Backbone(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)
        self.conv_out6 = nn.Conv2d(2048, 256, kernel_size=3, padding=1, stride=2)
        self.conv_out7 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)
        self.prj_5 = nn.Conv2d(2048, 256, kernel_size=1)
        self.prj_4 = nn.Conv2d(1024, 256, kernel_size=1)
        self.prj_3 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv_5 =nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv_4 =nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv_3 =nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # head =======================================================
        self.conv_cls = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.classes, kernel_size=3, padding=1))
        self.conv_reg = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4, kernel_size=3, padding=1))

        # reinit head =======================================================
        for layer in self.conv_cls.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.constant_(layer.bias, 0)
                nn.init.normal_(layer.weight, mean=0, std=0.01)
        for layer in self.conv_reg.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.constant_(layer.bias, 0)
                nn.init.normal_(layer.weight, mean=0, std=0.01)
        pi = 0.01
        _bias = -math.log((1.0-pi)/pi)
        nn.init.constant_(self.conv_cls[-1].bias, _bias)

        # learnable parameter for scales =====================================
        self.scale_div = nn.Parameter(torch.ones(len(self.regions) - 1))

        # generate anchors =================================================== 
        self._a_center_yx, self._a_tlbr_max_minmax, self._a_center_offset_max = \
            gen_anchors(self.view_size, self.first_stride, self.regions, self.center_offset_ratio)
        self.a_shw = self._a_center_yx.shape[0]
        self.register_buffer('a_center_yx', self._a_center_yx)
        self.register_buffer('a_tlbr_max_minmax', self._a_tlbr_max_minmax)
        self.register_buffer('a_center_offset_max', self._a_center_offset_max)


    def upsample(self, input):
        return F.interpolate(input, size=(input.shape[2]*2-1, input.shape[3]*2-1),
                    mode='bilinear', align_corners=True) # input size must be odd
    

    def forward(self, x, loc, label_class=None, label_box=None):
        '''
        Param:
        x:           FloatTensor(batch_num, 3, H, W)
        loc:         FloatTensor(batch_num, 4)
        label_class: LongTensor (batch_num, N_max) or None
        label_box:   FloatTensor(batch_num, N_max, 4) or None

        Return 1:
        loss: FloatTensor(batch_num)

        Return 2:
        cls_i_preds: LongTensor (batch_num, topk)
        cls_p_preds: FloatTensor(batch_num, topk)
        reg_preds:   FloatTensor(batch_num, topk, 4)
        '''
        
        C3, C4, C5 = self.backbone(x)
        P5 = self.prj_5(C5)
        P4 = self.prj_4(C4)
        P3 = self.prj_3(C3)
        P4 = P4 + self.upsample(P5)
        P3 = P3 + self.upsample(P4)
        P3 = self.conv_3(P3)
        P4 = self.conv_4(P4)
        P5 = self.conv_5(P5)
        P6 = self.conv_out6(C5)
        P7 = self.conv_out7(self.relu(P6))
        pred_list = [P3, P4, P5, P6, P7]
        assert len(pred_list) == len(self.regions) - 1
        cls_out = []
        reg_out = []
        for i, item in enumerate(pred_list):
            cls_i = self.conv_cls(item)
            reg_i = (self.conv_reg(item) * self.scale_div[i]).exp()
            cls_i = cls_i.permute(0,2,3,1).contiguous()
            reg_i = reg_i.permute(0,2,3,1).contiguous()
            cls_i = cls_i.view(cls_i.shape[0], -1, self.classes)
            reg_i = reg_i.view(reg_i.shape[0], -1, 4)
            # cls_i: [b, classes, H, W] -> [b, H*W, classes]
            # reg_i: [b, 4, H, W] -> [b, H*W, 4]
            cls_out.append(cls_i)
            reg_out.append(reg_i)
        cls_out = torch.cat(cls_out, dim=1)
        reg_out = torch.cat(reg_out, dim=1)
        # cls_out[b, shw, classes]
        # reg_out[b, shw, 4]
        if (label_class is not None) and (label_box is not None):
            targets_cls, targets_reg = self._encode(label_class, label_box, loc) 
            mask_cls = targets_cls > -1 # (b, shw)
            mask_reg = targets_cls > 0  # (b, shw)
            num_pos = torch.sum(mask_reg, dim=1).clamp_(min=1) # (b)
            loss = []
            for b in range(targets_cls.shape[0]):
                reg_out[b] = distance2bbox(self.a_center_yx, reg_out[b])
                targets_reg[b] = distance2bbox(self.a_center_yx, targets_reg[b])
                cls_out_b = cls_out[b][mask_cls[b]] # (S+-, classes)
                reg_out_b = reg_out[b][mask_reg[b]] # (S+, 4)
                targets_cls_b = targets_cls[b][mask_cls[b]] # (S+-)
                targets_reg_b = targets_reg[b][mask_reg[b]] # (S+, 4)
                loss_cls_b = sigmoid_focal_loss(cls_out_b, targets_cls_b, 2.0, 0.25).sum().view(1)
                loss_reg_b = iou_loss(reg_out_b, targets_reg_b).sum().view(1)
                loss.append((loss_cls_b + loss_reg_b) / float(num_pos[b])) 
            return torch.cat(loss, dim=0) # (b)
        else:
            return self._decode(cls_out, reg_out, loc)


    def _encode(self, label_class, label_box, loc):
        '''
        Param:
        label_class: LongTensor(batch_num, N_max)    bg:0
        label_box:   FloatTensor(batch_num, N_max, 4)
        loc:         FloatTensor(batch_num, 4)

        Return:
        targets_cls: LongTensor(batch_num, shw)
        targets_reg: FloatTensor(batch_num, shw, 4)
        '''
        targets_cls, targets_reg = [], []
        for b in range(label_class.shape[0]):
            # get tlbr
            tl = self.a_center_yx[:, None, :] - label_box[b][:, :2] 
            br = label_box[b][:, 2:] - self.a_center_yx[:, None, :]
            tlbr = torch.cat([tl, br], dim=2) # (shw, N_max, 4)
            # get center_offset
            label_yx = (label_box[b, :, :2] + label_box[b, :, 2:]) / 2.0 # (N_max, 2)
            center_offset_yx = self.a_center_yx[:, None, :] - label_yx[:, :] # (shw, N_max, 2)
            center_offset = (center_offset_yx[:, :, 0]**2 + 
                                center_offset_yx[:, :, 1]**2).sqrt() # (shw, N_max)
            # get mask_pos
            _min = torch.min(tlbr, dim=2)[0] # (shw, N_max)
            _max = torch.max(tlbr, dim=2)[0] # (shw, N_max)
            mask_inside = _min > 0 # (shw, N_max)
            mask_scale = (_max > self.a_tlbr_max_minmax[:, None, 0]) \
                & (_max < self.a_tlbr_max_minmax[:, None, 1]) # (shw, N_max)
            mask_center_offset = center_offset < \
                    self.a_center_offset_max.view(-1,1).expand_as(center_offset) # (shw, N_max)
            mask_pos = mask_inside & mask_scale & mask_center_offset # (shw, N_max)
            # get area
            area = (tlbr[:, :, 0] + tlbr[:, :, 2]) \
                * (tlbr[:, :, 1] + tlbr[:, :, 3]) # (shw, N_max)
            area[~mask_pos] = 999999999 # inf
            # get area_min_index
            area_min_index = torch.min(area, dim=1)[1] # (shw) long
            # get targets_cls_b
            targets_cls_b = label_class[b][area_min_index] # (shw)
            mask_pos_shw = torch.max(mask_pos, dim=1)[0]
            targets_cls_b[~mask_pos_shw] = 0
            # get targets_reg_b
            targets_reg_b = tlbr[ # tlbr: (shw, N_max, 4)
                torch.zeros_like(area, # (shw, N_max)
                 dtype=torch.uint8).scatter_(
                     1, area_min_index.view(-1,1), 1)] # (shw, 4)
            # ignore
            cd1 = self.a_center_yx - loc[b, :2]
            cd2 = loc[b, 2:] - self.a_center_yx
            mask = (cd1.min(dim=1)[0] < 0) | (cd2.min(dim=1)[0] < 0)
            targets_cls_b[mask] = -1
            # append
            targets_cls.append(targets_cls_b)
            targets_reg.append(targets_reg_b)
        return torch.stack(targets_cls), torch.stack(targets_reg)
    

    def _decode(self, cls_out, reg_out, loc, pred_acc=True):
        '''
        Param:
        cls_out: FloatTensor(batch_num, shw, classes)
        reg_out: FloatTensor(batch_num, shw, 4)
        loc:     FloatTensor(batch_num, 4)
        
        Return:
        cls_i_preds: LongTensor(batch_num, topk)
        cls_p_preds: FloatTensor(batch_num, topk)
        reg_preds:   FloatTensor(batch_num, topk, 4)
        '''
        cls_p_preds, cls_i_preds = torch.max(cls_out.sigmoid(), dim=2) # (b, shw)
        if pred_acc:
            cls_i_preds = cls_i_preds + 1
        reg_preds = []
        for b in range(cls_out.shape[0]):
            # box transform
            reg_tlbr = reg_out[b] # (shw, 4)
            reg_yxyx = distance2bbox(self.a_center_yx, reg_tlbr) # (shw, 4)
            reg_preds.append(reg_yxyx)
            # ignore
            cd1 = self.a_center_yx - loc[b, :2]
            cd2 = loc[b, 2:] - self.a_center_yx
            mask = (cd1.min(dim=1)[0] < 0) | (cd2.min(dim=1)[0] < 0)
            cls_p_preds[b, mask] = 0
        reg_preds = torch.stack(reg_preds) # (b, shw)
        # topk
        nms_maxnum = min(int(self.max_detections), int(cls_p_preds.shape[1]))
        select = torch.topk(cls_p_preds, nms_maxnum, largest=True, dim=1)[1]
        _cls_i, _cls_p, _reg = [], [], []
        for b in range(cls_out.shape[0]):
            _cls_i.append(cls_i_preds[b][select[b]]) # (topk)
            _cls_p.append(cls_p_preds[b][select[b]]) # (topk)
            reg_preds_b = reg_preds[b][select[b]] # (topk, 4)
            reg_preds_b[:, 0].clamp_(min=float(loc[b, 0]))
            reg_preds_b[:, 1].clamp_(min=float(loc[b, 1]))
            reg_preds_b[:, 2].clamp_(max=float(loc[b, 2]))
            reg_preds_b[:, 3].clamp_(max=float(loc[b, 3]))
            _reg.append(reg_preds_b) # (topk, 4)
        return torch.stack(_cls_i), torch.stack(_cls_p), torch.stack(_reg)



def get_loss(temp):
    return torch.mean(temp)



def get_pred(temp, nms_th, nms_iou):
    '''
    temp:
    cls_i_preds: LongTensor(batch_num, topk)
    cls_p_preds: FloatTensor(batch_num, topk)
    reg_preds:   FloatTensor(batch_num, topk, 4)

    Return:
    cls_i_preds: (LongTensor(s1), LongTensor(s2), ...)
    cls_p_preds: (FloatTensor(s1), FloatTensor(s2), ...)
    reg_preds:   (FloatTensor(s1,4), FloatTensor(s2,4), ...)
    '''
    cls_i_preds, cls_p_preds, reg_preds = temp
    _cls_i_preds, _cls_p_preds, _reg_preds = [], [], []
    for b in range(cls_i_preds.shape[0]):
        cls_i_preds_b = cls_i_preds[b]
        cls_p_preds_b = cls_p_preds[b]
        reg_preds_b = reg_preds[b]
        mask = cls_p_preds_b > nms_th
        cls_i_preds_b = cls_i_preds_b[mask]
        cls_p_preds_b = cls_p_preds_b[mask]
        reg_preds_b = reg_preds_b[mask]
        keep = box_nms(reg_preds_b, cls_p_preds_b, nms_iou)
        _cls_i_preds.append(cls_i_preds_b[keep])
        _cls_p_preds.append(cls_p_preds_b[keep])
        _reg_preds.append(reg_preds_b[keep])
    return _cls_i_preds, _cls_p_preds, _reg_preds
