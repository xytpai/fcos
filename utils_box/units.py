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
    targets_reg:    FloatTensor(acc_scale(Hi*Wi), 4)
    '''
    tl = centre_yx[:, None, :] - label_box[:, 0:2] 
    br = label_box[:, 2:4] - centre_yx[:, None, :]
    tlbr = torch.cat([tl, br], dim=2) # (ac, N, 4)
    _min = torch.min(tlbr, dim=2)[0]
    _max = torch.max(tlbr, dim=2)[0] # (ac, N)
    _max_min_index = torch.min(_max, dim=1)[1] # (ac)
    mask_inside = _min > 0 # (ac, N)
    mask_scale = (_max>centre_minmax[:,None,0])&(_max<=centre_minmax[:,None,1])
    targets_cls = torch.
    print(_max_min_index)
    # print(_max)
    # print(centre_minmax)
    # print(mask_scale)
    # print(l[:, 0, :])



if __name__ == '__main__':

    img_size = 5
    first_stride = 2
    regions = [0, 4, 8]
    centre_yx, centre_minmax = get_centre(img_size, first_stride, regions)
    print(centre_yx)
    # print(centre_minmax)
    label_class = torch.LongTensor([2, 4, 0])
    label_box = torch.FloatTensor([
        [1,1,2,2],
        [4,5,6,8],
        [2,2,2,2]
    ])
    targets_cls, targets_reg = get_output(centre_yx, centre_minmax,
        label_class, label_box)


