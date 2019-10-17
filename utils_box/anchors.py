import torch



def gen_anchors(img_size, first_stride, 
        tlbr_max_regions, center_offset_ratio=1.5):
    '''
    Return:
    center_yx:            FloatTensor(acc_scale(Hi*Wi), 2)
    tlbr_max_minmax:      FloatTensor(acc_scale(Hi*Wi), 2)
    center_offset_max:    FloatTensor(acc_scale(Hi*Wi))

    Note:
    - Hi,Wi accumulate from big to small
    
    Example:
    img_size = 641 
    first_stride = 8  # the stride of smallest anchors
    tlbr_max_regions = [0, 64, 128, 256, 512, 9999] # 5 scales
    center_offset_ratio = 1.5
    '''
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    scales = len(tlbr_max_regions) - 1
    center_yx, tlbr_max_minmax, center_offset_max = [], [], []
    stride = first_stride
    for scale_id in range(scales):
        fsz_h = (img_size[0]-1) // (first_stride * pow(2, scale_id)) + 1
        fsz_w = (img_size[1]-1) // (first_stride * pow(2, scale_id)) + 1
        center_yx_i = torch.zeros(fsz_h, fsz_w, 2)
        for h in range(fsz_h):
            for w in range(fsz_w):
                c_y, c_x = h * float(stride), w * float(stride)
                center_yx_i[h, w, :] = torch.Tensor([c_y, c_x])
        center_yx_i = center_yx_i.view(fsz_h*fsz_w, 2)
        tlbr_max_minmax_i = torch.zeros(fsz_h*fsz_w, 2)
        tlbr_max_minmax_i[:, 0] = tlbr_max_regions[scale_id]
        tlbr_max_minmax_i[:, 1] = tlbr_max_regions[scale_id + 1]
        center_offset_max_i = torch.zeros(fsz_h*fsz_w)
        center_offset_max_i[:] = float(stride) * center_offset_ratio
        stride *= 2
        center_yx.append(center_yx_i)
        tlbr_max_minmax.append(tlbr_max_minmax_i)
        center_offset_max.append(center_offset_max_i)
    return torch.cat(center_yx, dim=0), \
            torch.cat(tlbr_max_minmax, dim=0), \
                torch.cat(center_offset_max, dim=0)



def distance2bbox(points, distance):
    '''
    Param:
    points:   FloatTensor(n, 2)  2: y x
    distance: FloatTensor(n, 4)  4: top left bottom right

    Return:
    FloatTensor(n, 4) 4: ymin xmin ymax xmax
    '''
    ymin = points[:, 0] - distance[:, 0]
    xmin = points[:, 1] - distance[:, 1]
    ymax = points[:, 0] + distance[:, 2]
    xmax = points[:, 1] + distance[:, 3]
    return torch.stack([ymin, xmin, ymax, xmax], -1)
