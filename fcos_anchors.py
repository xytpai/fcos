import torch



def gen_anchors(img_size, first_stride, regions):
    '''
    Return:
    center_yx:      FloatTensor(acc_scale(Hi*Wi), 2)
    center_minmax:  FloatTensor(acc_scale(Hi*Wi), 2)

    Note:
    - Hi,Wi accumulate from big to small
    
    Example:
    img_size = 641 
    first_stride = 8  # the stride of smallest anchors
    regions = [0, 64, 128, 256, 512, 9999] # 5 scales
    '''
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    scales = len(regions) - 1
    center_yx, center_minmax = [], []
    stride = first_stride
    for scale_id in range(scales):
        fsz_h = (img_size[0]-1) // (first_stride * pow(2, scale_id)) + 1
        fsz_w = (img_size[1]-1) // (first_stride * pow(2, scale_id)) + 1
        center_yx_i = torch.zeros(fsz_h, fsz_w, 2)
        center_minmax_i = torch.zeros(fsz_h*fsz_w, 2)
        center_minmax_i[:, 0] = regions[scale_id]
        center_minmax_i[:, 1] = regions[scale_id + 1]
        for h in range(fsz_h):
            for w in range(fsz_w):
                c_y, c_x = h * float(stride), w * float(stride)
                center_yx_i[h, w, :] = torch.Tensor([c_y, c_x])
        stride *= 2
        center_yx_i = center_yx_i.view(fsz_h*fsz_w, 2)
        center_yx.append(center_yx_i)
        center_minmax.append(center_minmax_i)
    return torch.cat(center_yx, dim=0), torch.cat(center_minmax, dim=0)

