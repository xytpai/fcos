import torch



def bbox_overlaps_aligned(bboxes1, bboxes2, is_aligned=False):
    '''
    Param:
    bboxes1:   FloatTensor(n, 4) # 4: ymin, xmin, ymax, xmax
    bboxes2:   FloatTensor(n, 4)

    Return:    
    FloatTensor(n)
    '''
    tl = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
    br = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]
    hw = (br - tl + 1).clamp(min=0)  # [rows, 2]
    overlap = hw[:, 0] * hw[:, 1]
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1] + 1)
    ious = overlap / (area1 + area2 - overlap)
    return ious



def iou_loss(pred, target, eps=1e-6):
    '''
    Param:
    pred:     FloatTensor(n, 4) # 4: ymin, xmin, ymax, xmax
    target:   FloatTensor(n, 4)

    Return:    
    FloatTensor(n)
    '''
    ious = bbox_overlaps_aligned(pred, target).clamp(min=eps)
    loss = -ious.log()
    return loss
