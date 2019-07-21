import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import imp


try:
    imp.find_module('sigmoid_focal_loss_cuda')
    import sigmoid_focal_loss_cuda

    print('use lib sigmoid_focal_loss_cuda')

    class SigmoidFocalLossFunction(Function):

        @staticmethod
        def forward(ctx, input, target, gamma=2.0, alpha=0.25):
            ctx.save_for_backward(input, target)
            num_classes = input.shape[1]
            ctx.num_classes = num_classes
            ctx.gamma = gamma
            ctx.alpha = alpha
            loss = sigmoid_focal_loss_cuda.forward(input, target, num_classes,
                                                gamma, alpha)
            return loss

        @staticmethod
        @once_differentiable
        def backward(ctx, d_loss):
            input, target = ctx.saved_tensors
            num_classes = ctx.num_classes
            gamma = ctx.gamma
            alpha = ctx.alpha
            d_loss = d_loss.contiguous()
            d_input = sigmoid_focal_loss_cuda.backward(input, target, d_loss,
                                                    num_classes, gamma, alpha)
            return d_input, None, None, None, None

    sigmoid_focal_loss = SigmoidFocalLossFunction.apply

except ImportError:

    def sigmoid_focal_loss(input, target, gamma=2.0, alpha=0.25):
        input = input.sigmoid()
        one_hot = torch.zeros(input.shape[0], 
                1 + input.shape[1]).to(input.device).scatter_(1, 
                    target.view(-1,1), 1)
        one_hot = one_hot[:, 1:]
        pt = input*one_hot + (1.0-input)*(1.0-one_hot)
        w = alpha*one_hot + (1.0-alpha)*(1.0-one_hot)
        w = w * torch.pow((1.0-pt), gamma)
        loss = -w * pt.log()
        return loss


if __name__ == '__main__':

    a = torch.FloatTensor([[0.2,0.5,-0.5,0.5],[0.2,-0.2,0.4,0.5],[0.2,-0.1,0.3,-0.5]]).cuda()
    b = torch.LongTensor([1,2,0]).cuda()
    loss = sigmoid_focal_loss(a, b)
    print(loss)