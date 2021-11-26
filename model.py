import torch

from torch.nn.utils import weight_norm as wn
from torch.nn import functional as F
from torch import nn
from utils import concat_elu
from layers import GatedResnet, Nin
from locally_masked_convolution import LocallyMaskedConv2d


class LMPCNNUp(nn.Module):
    """
    Sequence of convolution layers with residual connections in the first half of the U-net.
    """

    def __init__(self, nr_resnet, nr_filters, conv_op, resnet_nonlinearity=concat_elu):
        super(LMPCNNUp, self).__init__()
        self.nr_resnet = nr_resnet
        self.u_stream = nn.ModuleList([GatedResnet(nr_filters, conv_op,
                                                   resnet_nonlinearity, skip_connection=0)
                                       for _ in range(nr_resnet)])

    def forward(self, u, mask=None):
        u_list = [layer(u, mask=mask) for layer in self.u_stream]
        return u_list


class LMPCNNDown(nn.Module):
    """
    Sequence of convolution layers with residual connections in the second half of the U-net, with
    residual connections to blocks in the first half.
    """

    def __init__(self, nr_resnet, nr_filters, conv_op, resnet_nonlinearity=concat_elu):
        super(LMPCNNDown, self).__init__()
        self.u_stream = nn.ModuleList([GatedResnet(nr_filters, conv_op,
                                                   resnet_nonlinearity, skip_connection=1)
                                       for _ in range(nr_resnet)])

    def forward(self, u, u_list, mask=None):
        for layer in self.u_stream:
            u = layer(u, a=u_list.pop(), mask=mask)

        return u


class LMPCNN(nn.Module):
    """
    Locally Masked Pixel Convolution Neural Networks.
    The model is structured as a U-net, with residual connections between the first and second halves.
    The blocks are connected with dilated convolutions, instead of downsampling/upsampling layers.
    Each block consists of multiple convolution layers with residual connections.
    The masks are used to enforce autoregressive properties, by masking the input to the convolution filters.
    """
    kernel_size = (5, 5)
    max_dilation = 2
    mask_weight = True

    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10, input_channels=3):
        super(LMPCNN, self).__init__()

        def conv_op_init(cin, cout):
            return wn(LocallyMaskedConv2d(cin, cout, kernel_size=self.kernel_size, mask_weight=self.mask_weight))

        def conv_op_dilated(cin, cout):
            return wn(LocallyMaskedConv2d(cin, cout, kernel_size=self.kernel_size, dilation=self.max_dilation,
                                          mask_weight=self.mask_weight))

        def conv_op(cin, cout):
            return wn(LocallyMaskedConv2d(cin, cout, kernel_size=self.kernel_size, mask_weight=self.mask_weight))

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([LMPCNNDown(down_nr_resnet[i], nr_filters, conv_op) for i in range(3)])

        self.up_layers = nn.ModuleList([LMPCNNUp(nr_resnet, nr_filters, conv_op) for _ in range(3)])

        self.downsize_u_stream = nn.ModuleList([conv_op_dilated(nr_filters, nr_filters) for _ in range(2)])
        self.upsize_u_stream = nn.ModuleList([conv_op_dilated(nr_filters, nr_filters) for _ in range(2)])

        self.u_init = conv_op_init(input_channels + 1, nr_filters)

        num_mix = 3 if input_channels == 1 else 10
        self.nin_out = Nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None

    def forward(self, x, mask_init, mask_undilated, mask_dilated, sample=False):
        if self.init_padding is None and not sample:
            padding_s = list(x.size())
            padding_s[1] = 1
            self.init_padding = torch.ones(padding_s, requires_grad=False, device=x.device)

        if sample:
            padding_s = list(x.size())
            padding_s[1] = 1
            padding = torch.ones(padding_s, requires_grad=False, device=x.device)
            x = torch.cat((x, padding), 1)

        x = x if sample else torch.cat((x, self.init_padding), 1)

        # Up pass
        u_list = [self.u_init(x, mask=mask_init)]
        for i in range(2):
            u_list += self.up_layers[i](u_list[-1], mask=mask_undilated)
            u_list += [self.downsize_u_stream[i](u_list[-1], mask=mask_dilated)]
        u_list += self.up_layers[2](u_list[-1], mask=mask_undilated)

        # Down pass
        u = u_list.pop()
        for i in range(2):
            u = self.down_layers[i](u, u_list, mask=mask_undilated)
            u = self.upsize_u_stream[i](u, mask=mask_dilated)
        u = self.down_layers[2](u, u_list, mask=mask_undilated)

        x_out = self.nin_out(F.elu(u))
        return x_out
