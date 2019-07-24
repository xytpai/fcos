import torch
import imp


try:
    imp.find_module('nms_cpu')
    import nms_cpu

    print('use lib nms_cpu')

    def box_nms(bboxes, scores, threshold=0.5):
        '''
        Param:
        bboxes: FloatTensor(n,4) # 4: ymin, xmin, ymax, xmax
        scores: FloatTensor(n)

        Return:
        keep:   LongTensor(s)
        '''
        scores = scores.view(-1, 1)
        bboxes_scores = torch.cat([bboxes, scores], dim=1) # (n, 5)
        keep = nms_cpu.nms(bboxes_scores, threshold) # (s)
        return keep

except ImportError:

    def box_nms(bboxes, scores, threshold=0.5):
        '''
        Param:
        bboxes: FloatTensor(n,4) # 4: ymin, xmin, ymax, xmax
        scores: FloatTensor(n)

        Return:
        keep:   LongTensor(s)
        '''
        mode = 'union'
        eps=1e-10
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
