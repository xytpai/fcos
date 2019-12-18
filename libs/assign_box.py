import torch
import imp


try:
    imp.find_module('assign_box_cuda')

    import assign_box_cuda

    print('use lib assign_box_cuda')


    def assign_box(label_cls, label_box, locs, im_h, im_w, ph, pw,
                    tlbr_max_min, tlbr_max_max, r):
        '''
        GPU >= 6.1

        Param:
        label_cls:  L(b, n_max)         0:bg  1~:fg, 0pad
        label_box:  F(b, n_max, 4)      ymin, xmin, ymax, xmax, 0:pad
        locs:       F(b, 4)             ymin, xmin, ymax, xmax

        im_h = 1025
        im_w = 1025
        ph = 129
        pw = 129

        tlbr_max_min = 5
        tlbr_max_max = 65
        r = 12

        Return:
        target_cls:  L(b, ph, pw)       -1:ign  0:bg  1~:fg
        target_box:  F(b, ph, pw, 4)    ymin, xmin, ymax, xmax

        -> F(b, ph, pw, 1 + 4)

        Note:
        n_max <= 200
        '''
        assert label_cls.dtype == torch.long
        assert label_box.dtype == torch.float
        assert locs.dtype == torch.float
        assert label_cls.is_cuda
        assert label_box.is_cuda
        assert locs.is_cuda
        b, n_max = label_cls.shape
        assert label_cls.shape == (b, n_max)
        assert label_box.shape == (b, n_max, 4)
        assert n_max <= 200
        output = assign_box_cuda.assign_box(label_cls, label_box, locs, im_h, im_w, ph, pw,
                                                tlbr_max_min, tlbr_max_max, r) # F(b, ph, pw, 1+4)
        target_cls, target_box = torch.split(output, [1, 4], dim=3)
        target_cls = target_cls.squeeze(3)
        # target_cls = assign_box_cuda.smooth(target_cls) # F(b, ph, pw)
        return target_cls.long(), target_box
    

except ImportError:

    raise 'not find lib assign_box_cuda'

