import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
import roi_align_cuda


class RoIAlignFunction(Function):
    '''
    Param:
    features:      FloatTensor(b, c, h, w)
    rois:          FloatTensor(num_rois, 5) b_index, ymin, xmin, ymax, xmax
    out_size:      int
    spatial_scale: float
    sample_num:    2 in paper

    Return:        FloatTensor(num_rois, c, out_size, out_size)
    '''

    @staticmethod
    def forward(ctx, features, rois, out_size, spatial_scale, sample_num=0):
        out_h, out_w = _pair(out_size)
        assert isinstance(out_h, int) and isinstance(out_w, int)
        ctx.spatial_scale = spatial_scale
        ctx.sample_num = sample_num
        ctx.save_for_backward(rois)
        ctx.feature_size = features.size()

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)

        output = features.new_zeros(num_rois, num_channels, out_h, out_w)
        if features.is_cuda:
            roi_align_cuda.forward(features, rois, out_h, out_w, spatial_scale,
                                   sample_num, output)
        else:
            raise NotImplementedError

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        feature_size = ctx.feature_size
        spatial_scale = ctx.spatial_scale
        sample_num = ctx.sample_num
        rois = ctx.saved_tensors[0]
        assert (feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = feature_size
        out_w = grad_output.size(3)
        out_h = grad_output.size(2)

        grad_input = grad_rois = None
        if ctx.needs_input_grad[0]:
            grad_input = rois.new_zeros(batch_size, num_channels, data_height,
                                        data_width)
            roi_align_cuda.backward(grad_output.contiguous(), rois, out_h,
                                    out_w, spatial_scale, sample_num,
                                    grad_input)

        return grad_input, grad_rois, None, None, None


roi_align = RoIAlignFunction.apply



if __name__ == '__main__':
    features = torch.Tensor([
        [
            [[1,2,3,4],
             [2,3,4,5],
             [3,4,5,6],
             [4,5,6,7]]
        ]
    ])
    ymin_1, xmin_1, ymax_1, xmax_1 = 0, 1, 2, 3
    ymin_2, xmin_2, ymax_2, xmax_2 = 0, 0, 2, 2
    roi = torch.Tensor([
        [0, ymin_1, xmin_1, ymax_1, xmax_1],
        [0, ymin_2, xmin_2, ymax_2, xmax_2],
    ])
    print(features)
    rois_f = roi_align(features.cuda(), roi.cuda(), 2, 1)
    print(rois_f.shape)
    print(rois_f)
    # tensor([[[[3.5000, 4.6875],
    #           [5.0000, 6.1875]]],
    #         [[[2.5000, 4.0000],
    #           [4.0000, 5.5000]]]], device='cuda:0')
    pass