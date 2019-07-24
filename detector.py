import math
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from libs.sigmoid_focal_loss import sigmoid_focal_loss
# TODO: choose backbone
from backbone import resnet50 as backbone



class Detector(nn.Module):
    def __init__(self, pretrained=False):
        super(Detector, self).__init__()

        # ---------------------------
        # TODO: Param
        self.regions = [0, 64, 128, 256, 512 ,1024]
        self.first_stride = 8
        self.train_size = 641
        self.eval_size = 641
        self.classes = 20
        self.nms = True
        self.nms_th = 0.05
        self.nms_iou = 0.5
        self.max_detections = 1000
        # ---------------------------

        self.backbone = backbone(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)
        self.conv_out6 = nn.Conv2d(2048, 256, kernel_size=3, padding=1, stride=2)
        self.conv_out7 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)
        self.prj_5 = nn.Conv2d(2048, 256, kernel_size=1)
        self.prj_4 = nn.Conv2d(1024, 256, kernel_size=1)
        self.prj_3 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv_5 =nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv_4 =nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv_3 =nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.scale_3 = nn.Parameter(torch.tensor(1.0).float())
        # self.scale_4 = nn.Parameter(torch.tensor(1.0).float())
        # self.scale_5 = nn.Parameter(torch.tensor(1.0).float())
        # self.scale_6 = nn.Parameter(torch.tensor(1.0).float())
        # self.scale_7 = nn.Parameter(torch.tensor(1.0).float())

        self.conv_cls = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1 + self.classes, kernel_size=3, padding=1))
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


    def upsample(self, input):
        '''
        ATTENTION: size must be odd
        '''
        return F.interpolate(input, size=(input.shape[2]*2-1, input.shape[3]*2-1),
                    mode='bilinear', align_corners=True)
    

    def forward(self, x, targets_cls=None, targets_cen=None, targets_reg=None):
        '''
        Param:
        targets_cls: LongTensor(b, sum_scale(Hi*Wi))
        targets_cen: FloatTensor(b, sum_scale(Hi*Wi))
        targets_reg: FloatTensor(b, sum_scale(Hi*Wi), 4)

        Return (not loss):
        cls_out: FloatTensor(b, sum_scale(Hi*Wi), 1 + classes)
        reg_out: FloatTensor(b, sum_scale(Hi*Wi), 4)

        Note:
        sum_scale(): [P(i), P(i+1), P(i+2), ...]
        '''
        C3, C4, C5 = self.backbone(x)
        
        P5 = self.prj_5(C5)
        P5_upsampled = self.upsample(P5)
        P5 = self.conv_5(P5)

        P4 = self.prj_4(C4)
        P4 = P5_upsampled + P4
        P4_upsampled = self.upsample(P4)
        P4 = self.conv_4(P4)

        P3 = self.prj_3(C3)
        P3 = P4_upsampled + P3
        P3 = self.conv_3(P3)

        P6 = self.conv_out6(C5)
        P7 = self.conv_out7(self.relu(P6))

        pred_list = [P3, P4, P5, P6, P7]
        # scale_list = [self.scale_3, self.scale_4, 
        #                 self.scale_5, self.scale_6, self.scale_7]

        cls_out = []
        reg_out = []
        for i in range(len(pred_list)):
            cls_i = self.conv_cls(pred_list[i])
            reg_i = self.conv_reg(pred_list[i])
            # reg_i = reg_i * scale_list[i]
            # cls_i: [b, 1 + classes, H, W] -> [b, H*W, 1 + classes]
            cls_i = cls_i.permute(0,2,3,1).contiguous()
            cls_i = cls_i.view(cls_i.shape[0], -1, 1 + self.classes)
            # reg_i: [b, 4, H, W] -> [b, H*W, 4]
            reg_i = reg_i.permute(0,2,3,1).contiguous()
            reg_i = reg_i.view(reg_i.shape[0], -1, 4)
            cls_out.append(cls_i)
            reg_out.append(reg_i)
        
        # cls_out[b, sum_scale(Hi*Wi), 1 + classes]
        # reg_out[b, sum_scale(Hi*Wi), 4]
        cls_out = torch.cat(cls_out, dim=1)
        reg_out = torch.cat(reg_out, dim=1)

        if targets_cls is None:
            return cls_out, reg_out
        else:
            cen_out = cls_out[:, :, 0] # (b, sum_scale(Hi*Wi))
            cls_out = cls_out[:, :, 1:]
            cls_out = cls_out.view(-1, self.classes)
            cen_out = cen_out.view(-1)
            reg_out = reg_out.view(-1, 4)
            targets_cls = targets_cls.view(-1)
            targets_reg = targets_reg.view(-1, 4)
            targets_cen = targets_cen.view(-1)
            loss_cls_1 = sigmoid_focal_loss(cls_out, targets_cls, 2.0, 0.25)
            loss_cls_1 = torch.sum(loss_cls_1).view(1)
            mask_reg = targets_cls > 0 # (S+)
            reg_out = reg_out[mask_reg] # (S+, 4)
            targets_reg = targets_reg[mask_reg] # # (S+, 4)
            cen_out = cen_out[mask_reg]
            targets_cen = targets_cen[mask_reg]
            loss_reg = F.smooth_l1_loss(reg_out, targets_reg, reduction='sum').view(1)
            loss_cen = F.binary_cross_entropy_with_logits(cen_out, targets_cen, reduction='sum').view(1)
            num_pos = torch.sum(mask_reg).view(1)
            return (loss_cls_1, loss_reg, loss_cen, num_pos)



def get_loss(temp):
    loss_cls_1, loss_reg, loss_cen, num_pos = temp
    loss_cls = torch.sum(loss_cls_1)
    loss_reg = torch.sum(loss_reg)
    loss_cen = torch.sum(loss_cen)
    num_pos = float(torch.sum(num_pos))
    if num_pos <= 0:
        num_pos = 1.0
    loss = (loss_cls + loss_reg + loss_cen) / num_pos
    return loss
