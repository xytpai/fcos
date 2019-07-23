import torch
import  utils_box.units as units
# TODO: define Encoder



def box_nms(bboxes, scores, threshold=0.5, mode='union', eps=1e-10):
    '''
    Param:
    bboxes: FloatTensor(n,4) # 4: ymin, xmin, ymax, xmax
    scores: FloatTensor(n)
    mode:   'union' or 'min'

    Return:
    LongTensor(S) # index of keep boxes
    '''
    ymin, xmin, ymax, xmax = bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,3]
    areas = (xmax-xmin+eps) * (ymax-ymin+eps)
    order = scores.sort(0, descending=True)[1]
    keep = []

    while order.numel() > 0:
        i = order[0] 
        keep.append(i)
        if order.numel() == 1:
            break
        _ymin = ymin[order[1:]].clamp(min=float(ymin[i]))
        _xmin = xmin[order[1:]].clamp(min=float(xmin[i]))
        _ymax = ymax[order[1:]].clamp(max=float(ymax[i]))
        _xmax = xmax[order[1:]].clamp(max=float(xmax[i]))
        _h = (_ymax-_ymin+eps).clamp(min=0)
        _w = (_xmax-_xmin+eps).clamp(min=0)
        inter = _h * _w
        if mode == 'union':
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == 'min':
            ovr = inter / areas[order[1:]].clamp(max=float(areas[i]))
        else:
            raise TypeError('Unknown nms mode: %s.' % mode)
        ids = (ovr<=threshold).nonzero().squeeze() + 1
        if ids.numel() == 0:
            break
        order = torch.index_select(order, 0, ids)
    return torch.LongTensor(keep)



class Encoder:
    def __init__(self, 
        regions = [0, 64, 128, 256, 512, 1024],
        first_stride = 8,
        train_size = 641, 
        eval_size = 641,
        nms = True, 
        nms_th = 0.05, 
        nms_iou = 0.5,
        max_detections = 1000):

        self.regions = regions
        self.first_stride = first_stride
        self.train_size = train_size
        self.eval_size = eval_size
        self.nms = nms
        self.nms_th = nms_th
        self.nms_iou = nms_iou
        self.max_detections = max_detections

        self.train_centre_yx, self.train_centre_minmax = \
            units.get_centre(self.train_size, self.first_stride, self.regions)
        self.eval_centre_yx, self.eval_centre_minmax = \
            units.get_centre(self.eval_size, self.first_stride, self.regions)


    def encode(self, label_class, label_box):
        '''
        Param:
        label_class: (LongTensor(N1), LongTensor(N2), ...)
        label_box:   (FloatTensor(N1,4), FloatTensor(N2,4), ...)

        Return:
        targets_cls: LongTensor(b, sum_scale(Hi*Wi))
        targets_cen: FloatTensor(b, sum_scale(Hi*Wi))
        targets_reg: FloatTensor(b, sum_scale(Hi*Wi), 4)
        

        Note:
        - class = 0 indicate background
        - in label_box 4 indicate ymin, xmin, ymax, xmax
        - all calculations are on the CPU
        - Hi,Wi accumulate from big to small
        - in targets_reg, 4 indicates:
            f1 -> log(top)
            f2 -> log(left)
            f3 -> log(bottom)
            f4 -> log(right)
        '''
        targets_cls = []
        targets_cen = []
        targets_reg = []

        for b in range(len(label_class)):

            if label_class[b].shape[0] == 0:
                targets_cls_b = torch.zeros(self.train_centre_yx.shape[0]).long()
                targets_cen_b = torch.zeros(self.train_centre_yx.shape[0])
                targets_reg_b = torch.zeros(self.train_centre_yx.shape[0], 4)
            else:
                targets_cls_b, targets_cen_b, targets_reg_b = units.get_output(
                    self.train_centre_yx, self.train_centre_minmax, 
                    label_class[b], label_box[b]) # (pos), (pos, 4)
                targets_reg_b = targets_reg_b.log()

            targets_cls.append(targets_cls_b)
            targets_cen.append(targets_cen_b)
            targets_reg.append(targets_reg_b)

        targets_cls = torch.stack(targets_cls, dim=0)
        targets_cen = torch.stack(targets_cen, dim=0)
        targets_reg = torch.stack(targets_reg, dim=0)

        return targets_cls, targets_cen, targets_reg


    def decode(self, cls_out, reg_out, scale_shift=None):
        '''
        Param:
        cls_out: FloatTensor(b, sum_scale(Hi*Wi), 1 + classes)
        reg_out: FloatTensor(b, sum_scale(Hi*Wi), 4)
        
        Return:
        if nms:
            cls_i_preds: (LongTensor(s1), LongTensor(s2), ...)
            cls_p_preds: (FloatTensor(s1), FloatTensor(s2), ...)
            reg_preds:   (FloatTensor(s1,4), FloatTensor(s2,4), ...)
        else:
            cls_i_preds: LongTensor(b, sum_scale(Hi*Wi))
            cls_p_preds: FloatTensor(b, sum_scale(Hi*Wi))
            reg_preds:   FloatTensor(b, sum_scale(Hi*Wi), 4)

        Note:
        - scale_shift: if not None, reg_preds /= float(scale_shift)
        - class = 0 indicate background
        - in reg_preds 4 indicate ymin, xmin, ymax, xmax
        - all calculations are on the CPU
        -  Hi,Wi accumulate from big to small
        - reg_out = f1, f2, f3, f4, decoding process:
            top    = f1.exp()
            left   = f2.exp()
            bottom = f3.exp()
            right  = f4.exp()
            ymin, xmin = cy - top, cx - left
            ymax, xmax = cy + bottom, cx + right
        '''
        cen_out = cls_out[:, :, 0] # (b, s)
        cls_out = cls_out[:, :, 1:] # (b, s, classes)
        cls_p_preds, cls_i_preds = torch.max(cls_out.sigmoid(), dim=2) # (b, s)
        cls_p_preds = cls_p_preds * cen_out.sigmoid()
        cls_i_preds = cls_i_preds + 1

        reg_preds = []
        for b in range(cls_out.shape[0]):
            tlbr = reg_out[b].exp() # (s, 4)
            tl, br = tlbr.split([2, 2], dim=1) # (s, 2)
            ymin_xmin = self.eval_centre_yx - tl
            ymax_xmax = self.eval_centre_yx + br
            ymin_xmin_ymax_xmax = torch.cat([ymin_xmin, ymax_xmax], dim=1)
            reg_preds.append(ymin_xmin_ymax_xmax)
        reg_preds = torch.stack(reg_preds, dim=0) # (b, s, 4)

        if self.nms == False:
            if scale_shift is not None:
                reg_preds /= float(scale_shift)
            return cls_i_preds, cls_p_preds, reg_preds
        
        # Topk
        nms_maxnum = min(int(self.max_detections), int(cls_p_preds.shape[1]))
        select = torch.topk(cls_p_preds, nms_maxnum, largest=True, dim=1)[1]

        # NMS
        _cls_i_preds = []
        _cls_p_preds = []
        _reg_preds = []

        for b in range(cls_out.shape[0]):

            cls_i_preds_b = cls_i_preds[b][select[b]] # (topk)
            cls_p_preds_b = cls_p_preds[b][select[b]] # (topk)
            reg_preds_b = reg_preds[b][select[b]] # (topk, 4)

            mask = cls_p_preds_b > self.nms_th
            cls_i_preds_b = cls_i_preds_b[mask]
            cls_p_preds_b = cls_p_preds_b[mask]
            reg_preds_b = reg_preds_b[mask]

            keep = box_nms(reg_preds_b, cls_p_preds_b, self.nms_iou)
            cls_i_preds_b = cls_i_preds_b[keep]
            cls_p_preds_b = cls_p_preds_b[keep]
            reg_preds_b = reg_preds_b[keep]

            reg_preds_b[:, :2] = reg_preds_b[:, :2].clamp(min=0)
            reg_preds_b[:, 2] = reg_preds_b[:, 2].clamp(max=self.eval_size-1)
            reg_preds_b[:, 3] = reg_preds_b[:, 3].clamp(max=self.eval_size-1)

            if scale_shift is not None:
                reg_preds_b /= float(scale_shift)

            _cls_i_preds.append(cls_i_preds_b)
            _cls_p_preds.append(cls_p_preds_b)
            _reg_preds.append(reg_preds_b)
            
        return _cls_i_preds, _cls_p_preds, _reg_preds
