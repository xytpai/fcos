import torch



def get_centre(img_size, first_stride, regions):
    '''
    Return:
    centre_yx:      FloatTensor(acc_scale(Hi*Wi), 2)
    centre_minmax:  FloatTensor(acc_scale(Hi*Wi), 2)

    Note:
    - Hi,Wi accumulate from big to small
    
    Example:
    img_size = 641 
    first_stride = 8  # the stride of smallest anchors
    regions = [0, 64, 128, 256, 512, 1024] # 5 scales
    '''

    scales = len(regions) - 1
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    
    centre_yx = []
    centre_minmax = []

    stride = first_stride

    for scale_id in range(scales):

        fsz_h = (img_size[0]-1) // (first_stride * pow(2, scale_id)) + 1
        fsz_w = (img_size[1]-1) // (first_stride * pow(2, scale_id)) + 1

        centre_yx_i = torch.zeros(fsz_h, fsz_w, 2)
        centre_minmax_i = torch.zeros(fsz_h*fsz_w, 2)

        centre_minmax_i[:, 0] = regions[scale_id]
        centre_minmax_i[:, 1] = regions[scale_id + 1]

        for h in range(fsz_h):
            for w in range(fsz_w):
                c_y, c_x = h * float(stride), w * float(stride)
                centre_yx_i[h, w, :] = torch.Tensor([c_y, c_x])

        stride *= 2

        centre_yx_i = centre_yx_i.view(fsz_h*fsz_w, 2)
        centre_yx.append(centre_yx_i)
        centre_minmax.append(centre_minmax_i)

    return torch.cat(centre_yx, dim=0), torch.cat(centre_minmax, dim=0)



def get_output(centre_yx, centre_minmax, label_class, label_box):
    '''
    Param:
    centre_yx:      FloatTensor(acc_scale(Hi*Wi), 2)
    centre_minmax:  FloatTensor(acc_scale(Hi*Wi), 2)
    label_class:    LongTensor(N)
    label_box:      FloatTensor(N, 4)

    Return:
    targets_cls:    LongTensor(acc_scale(Hi*Wi))
    targets_cen:    FloatTensor(acc_scale(Hi*Wi))
    targets_reg:    FloatTensor(acc_scale(Hi*Wi), 4)
    
    Note:
    in label_box 4: ymin, ymax, xmin, xmax
    in targets_reg 4: top, left, bottom, right
    '''
    eps = 1e-3

    tl = centre_yx[:, None, :] - label_box[:, 0:2] 
    br = label_box[:, 2:4] - centre_yx[:, None, :]
    tlbr = torch.cat([tl, br], dim=2) # (ac, N, 4)

    _min = torch.min(tlbr, dim=2)[0] # (ac, N)
    _max = torch.max(tlbr, dim=2)[0] # (ac, N)
    mask_inside = _min > 0 # (ac, N)
    mask_scale = (_max>centre_minmax[:,None,0])&(_max<=centre_minmax[:,None,1]) # (ac, N)
    neg_mask = ~torch.max((mask_inside&mask_scale), dim=1)[0] # (ac)

    _max[~mask_inside] = 1e8
    _max_min_index = torch.min(_max, dim=1)[1] # (ac)
    
    targets_cls = label_class[_max_min_index] # (ac)
    targets_cls[neg_mask] = 0

    _label_box = label_box[_max_min_index]
    _tl = centre_yx[:, :] - _label_box[:, 0:2] # (ac, 2)
    _br = _label_box[:, 2:4] - centre_yx[:, :] # (ac, 2)
    _tlbr = torch.cat([_tl, _br], dim=1) # (ac, 4)

    targets_reg = _tlbr
    targets_reg[neg_mask] = 1

    _lr = _tlbr[:, 1::2] # (ac, 2)
    _tb = _tlbr[:, 0::2] # (ac, 2)
    min_lr = torch.min(_lr, dim=1)[0] # (ac)
    max_lr = torch.max(_lr, dim=1)[0] # (ac)
    min_tb = torch.min(_tb, dim=1)[0] # (ac)
    max_tb = torch.max(_tb, dim=1)[0] # (ac)
    targets_cen = ((min_lr*min_tb+eps)/(max_lr*max_tb+eps)).sqrt()

    return targets_cls, targets_cen, targets_reg



if __name__ == '__main__':

    img_size = 5
    first_stride = 2
    regions = [0, 2, 8]
    centre_yx, centre_minmax = get_centre(img_size, first_stride, regions)
    # print(centre_yx)
    print(centre_minmax)
    label_class = torch.LongTensor([2, 4, 1])
    label_box = torch.FloatTensor([
        [1,1,2,2], # N
        [1,1,3,3], # Y
        [1,1,5,5] # Y
    ])
    targets_cls, targets_cen, targets_reg = get_output(
        centre_yx, centre_minmax, label_class, label_box)
    print(targets_cls)
    print(targets_cen)
    print(targets_reg)

