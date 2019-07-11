import os
import torch
import numpy as np
from collections import defaultdict



def bbox_iou_np(bbox_a, bbox_b):
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i + 1e-10)



def gen_prec_rec_np(
        pred_bboxes, pred_labels, pred_scores, 
        gt_bboxes, gt_labels, 
        iou_th=0.5):
    '''
    Param:
    pred_bboxes: [ndarray(N1,4), ndarray(N2,4), ...] # float, 4:ymin,xmin,ymax,xmax
    pred_labels: [ndarray(N1), ndarray(N2), ...]     # long,  0:background
    pred_scores: [ndarray(N1), ndarray(N2), ...]     # float
    gt_bboxes:   [ndarray(M1,4), ndarray(M2,4), ...]
    gt_labels:   [ndarray(M1), ndarray(M2), ...]

    Return:
    prec         list(C) # C:1+classes
    rec          list(C)

    Note:
    - If some categories do not appear, the corresponding prec,rec is None.
    - prec[0] = rec[0] == None
    '''
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    n_frames = len(gt_labels)

    for n in range(n_frames):

        pred_bbox  = pred_bboxes[n]
        pred_label = pred_labels[n]
        pred_score = pred_scores[n]
        gt_bbox    = gt_bboxes[n]
        gt_label   = gt_labels[n]

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):

            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]

            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]

            n_pos[l] += gt_mask_l.sum()
            score[l].extend(pred_score_l)

            if len(pred_score_l) == 0:
                continue

            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue
            
            # In order to prevent the area from being zero
            pred_bbox_l = pred_bbox_l.copy()
            gt_bbox_l = gt_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l[:, 2:] += 1

            iou = bbox_iou_np(pred_bbox_l, gt_bbox_l) # arr(N,M)
            gt_index = iou.argmax(axis=1) # arr(N)

            # mark -1 if IOU<th of a prediction box and the corresponding largest label box
            gt_index[iou.max(axis=1) < iou_th] = -1 
            del iou

            # selec indicates whether a label box has been selected
            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool) # arr(M)

            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if not selec[gt_idx]:
                        match[l].append(1)
                    else:
                        match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    n_fg_class = max(n_pos.keys()) + 1

    prec = [None]*n_fg_class
    rec  = [None]*n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)
        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        # according to different threshold
        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # if tp+fp == 0 the result is inf, is impossible
        prec[l] = tp / (fp + tp)

        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]
    return prec, rec



def gen_ap_np(prec, rec):
    '''
        1
    prec| .
        |    .
        |      .
        |       .
        |        .
       0 ———————————1
                   rec
    
    Return:
    ap: ndarray(C) # C=1+classes

    Note:
    - index 0 indicates background
    - ap[0] == np.nan
    '''
    n_class = len(prec)
    ap = np.empty(n_class) # np.float
    for l in range(n_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue
        
        # place sentinel point
        mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
        mrec = np.concatenate(([0], rec[l], [1]))

        # mpre =  np.array([8,2,1,3,4,1])
        # np.maximum.accumulate(mpre[::-1])[::-1]
        # return: np.array([8,4,4,4,4,1])
        mpre = np.maximum.accumulate(mpre[::-1])[::-1]

        # capture the index of changes in recall
        # mrec = np.array([1,2,3,3,5,6])
        # np.where(mrec[1:] != mrec[:-1])[0]
        # return: array([0,1,3,4], dtype=int64)
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # (\Delta recall) * prec
        ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap



def eval_detection(
    pred_bboxes, pred_labels, pred_scores, 
    gt_bboxes, gt_labels,
    iou_th=0.5):
    '''
    Param:
    pred_bboxes: [ndarray(N1,4), ndarray(N2,4), ...] # float, 4:ymin,xmin,ymax,xmax
    pred_labels: [ndarray(N1), ndarray(N2), ...]     # long,  0:background
    pred_scores: [ndarray(N1), ndarray(N2), ...]     # float
    gt_bboxes:   [ndarray(M1,4), ndarray(M2,4), ...]
    gt_labels:   [ndarray(M1), ndarray(M2), ...]

    Return:
    out['ap']:   ndarray(C) # C=1+classes
    out['map']   float

    Note:
    - index 0 indicates background
    - ap[0] == np.nan
    '''
    prec, rec = gen_prec_rec_np(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels,
        iou_th=iou_th)
    ap = gen_ap_np(prec, rec)
    return {'ap': ap, 'map': np.nanmean(ap)}
