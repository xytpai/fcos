import math
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from libs.assign_box import assign_box
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
        self.view_size = 1025 # odd
        self.classes = 80
        self.nms_th = 0.05
        self.nms_iou = 0.6
        self.max_detections = 3000
        self.tlbr_max_minmax = [[5, 64], [64, 128], [128, 256], [256, 512], [512, 1024]]
        self.phpw = [[129, 129], [65, 65], [33, 33], [17, 17], [9, 9]]
        self.r = [12, 24, 48, 96, 192]
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
        init_layers = [self.conv_cls, self.conv_reg]
        for init_layer in init_layers:
            for item in init_layer.children():
                if isinstance(item, nn.Conv2d) or isinstance(item, nn.ConvTranspose2d):
                    nn.init.constant_(item.bias, 0)
                    nn.init.normal_(item.weight, mean=0, std=0.01)
        pi = 0.01
        _bias = -math.log((1.0-pi)/pi)
        nn.init.constant_(self.conv_cls[-1].bias, _bias)

        # learnable parameter for scales =====================================
        self.scale_param = nn.Parameter(torch.ones(len(self.r)))
    

    def upsample(self, input):
        return F.interpolate(input, size=(input.shape[2]*2-1, input.shape[3]*2-1),
                    mode='bilinear', align_corners=True) # input size must be odd
    

    def decode_box(self, pred_box, im_h, im_w, ph, pw):
        '''
        Param:
        pred_box:  F(b, ph, pw, 4) top, left, bottom, right
        
        Return:
        pred_box:  F(b, ph, pw, 4) ymin, xmin, ymax, xmax
        '''
        y = torch.linspace(0, im_h-1, ph).to(pred_box.device)
        x = torch.linspace(0, im_w-1, pw).to(pred_box.device)
        center_y, center_x = torch.meshgrid(y, x) # F(ph, pw)
        center_y = center_y.squeeze(0)
        center_x = center_x.squeeze(0)
        ymin = center_y - pred_box[:, :, :, 0]
        xmin = center_x - pred_box[:, :, :, 1]
        ymax = center_y + pred_box[:, :, :, 2]
        xmax = center_x + pred_box[:, :, :, 3]
        return torch.stack([ymin, xmin, ymax, xmax], dim=3)
    

    def forward(self, imgs, locs, label_class=None, label_box=None):
        '''
        Param:
        imgs:        F(b, 3, vsz, vsz)
        locs:        F(b, 4)
        label_class: L(b, N_max) or None
        label_box:   F(b, N_max, 4) or None

        Return 1:
        loss:        F(b)

        Return 2:
        pred_cls_i:  L(b, topk)
        pred_cls_p:  F(b, topk)
        pred_reg:    F(b, topk, 4)
        '''
        
        # forward fpn
        C3, C4, C5 = self.backbone(imgs)
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
        assert len(pred_list) == len(self.r)

        # get pred
        pred_cls = []
        pred_reg = []
        for i, feature in enumerate(pred_list):
            cls_i = self.conv_cls(feature)
            reg_i =  (self.conv_reg(feature) * self.scale_param[i]).exp()
            cls_i = cls_i.permute(0,2,3,1).contiguous()
            reg_i = reg_i.permute(0,2,3,1).contiguous() # b, ph, pw, 4
            reg_i = self.decode_box(reg_i, self.view_size, self.view_size, self.phpw[i][0], self.phpw[i][1])
            pred_cls.append(cls_i.view(cls_i.shape[0], -1, self.classes))
            pred_reg.append(reg_i.view(reg_i.shape[0], -1, 4))
        pred_cls = torch.cat(pred_cls, dim=1)
        pred_reg = torch.cat(pred_reg, dim=1)
        # pred_cls: F(b, n, classes)
        # pred_reg: F(b, n, 4)

        if (label_class is not None) and (label_box is not None):
            # <= 200
            n_max = min(label_class.shape[1], 200)
            if n_max == 200:
                label_class = label_class[:, :200]
                label_box   = label_box[:, :200, :]
            # get target
            target_cls = []
            target_reg = []
            for i in range(len(self.r)):
                target_cls_i, target_reg_i = assign_box(label_class, label_box, locs,
                    self.view_size, self.view_size, self.phpw[i][0], self.phpw[i][1],
                    self.tlbr_max_minmax[i][0], self.tlbr_max_minmax[i][1], self.r[i])
                target_cls.append(target_cls_i.view(target_cls_i.shape[0], -1))
                target_reg.append(target_reg_i.view(target_reg_i.shape[0], -1, 4))
            target_cls = torch.cat(target_cls, dim=1) # L(b, n)
            target_reg = torch.cat(target_reg, dim=1) # F(b, n, 4)
            # get loss
            m_negpos = target_cls > -1 # B(b, n)
            m_pos    = target_cls > 0  # B(b, n)
            num_pos = torch.sum(m_pos, dim=1).clamp_(min=1) # L(b)
            loss = []
            for b in range(locs.shape[0]):
                pred_cls_b = pred_cls[b][m_negpos[b]]     # F(S+-, classes)
                target_cls_b = target_cls[b][m_negpos[b]] # L(S+-)
                pred_reg_b = pred_reg[b][m_pos[b]]        # F(S+, 4)
                target_reg_b = target_reg[b][m_pos[b]]    # F(S+, 4)
                loss_cls_b = sigmoid_focal_loss(pred_cls_b, target_cls_b, 2.0, 0.25).sum().view(1)
                loss_reg_b = iou_loss(pred_reg_b, target_reg_b).sum().view(1)
                loss.append((loss_cls_b + loss_reg_b) / float(num_pos[b]))
            return torch.cat(loss, dim=0) # F(b)
        else:
            return self._decode(pred_cls, pred_reg, locs)
    

    def _decode(self, pred_cls, pred_reg, locs):
        '''
        Param:
        pred_cls:   F(b, n, classes)
        pred_reg:   F(b, n, 4)
        locs:       F(b, 4)
        
        Return:
        pred_cls_i: L(b, topk)
        pred_cls_p: F(b, topk)
        pred_reg:   F(b, topk, 4)
        '''
        pred_cls_p, pred_cls_i = torch.max(pred_cls.sigmoid(), dim=2) # (b, n)
        pred_cls_i = pred_cls_i + 1
        # topk
        nms_maxnum = min(int(self.max_detections), int(pred_cls_i.shape[1]))
        select = torch.topk(pred_cls_p, nms_maxnum, largest=True, dim=1)[1]
        _cls_i, _cls_p, _reg = [], [], []
        for b in range(locs.shape[0]):
            _cls_i.append(pred_cls_i[b][select[b]]) # (topk)
            _cls_p.append(pred_cls_p[b][select[b]]) # (topk)
            pred_reg_b = pred_reg[b][select[b]] # (topk, 4)
            pred_reg_b[:, 0].clamp_(min=float(locs[b, 0]))
            pred_reg_b[:, 1].clamp_(min=float(locs[b, 1]))
            pred_reg_b[:, 2].clamp_(max=float(locs[b, 2]))
            pred_reg_b[:, 3].clamp_(max=float(locs[b, 3]))
            _reg.append(pred_reg_b) # (topk, 4)
        return torch.stack(_cls_i), torch.stack(_cls_p), torch.stack(_reg)



def get_loss(temp):
    return torch.mean(temp)



def get_pred(temp, nms_th, nms_iou):
    '''
    temp:
    pred_cls_i: L(b, topk)
    pred_cls_p: F(b, topk)
    pred_reg:   F(b, topk, 4)

    Return:
    cls_i_preds: (L(s1), L(s2), ...)
    cls_p_preds: (F(s1), F(s2), ...)
    reg_preds:   (F(s1,4), F(s2,4), ...)
    '''
    pred_cls_i, pred_cls_p, pred_reg = temp
    _pred_cls_i, _pred_cls_p, _pred_reg = [], [], []
    for b in range(pred_cls_i.shape[0]):
        m = pred_cls_p[b] > nms_th
        pred_cls_i_b = pred_cls_i[b][m]
        pred_cls_p_b = pred_cls_p[b][m]
        pred_reg_b = pred_reg[b][m]
        keep = box_nms(pred_reg_b, pred_cls_p_b, nms_iou)
        _pred_cls_i.append(pred_cls_i_b[keep])
        _pred_cls_p.append(pred_cls_p_b[keep])
        _pred_reg.append(pred_reg_b[keep])
    return _pred_cls_i, _pred_cls_p, _pred_reg
