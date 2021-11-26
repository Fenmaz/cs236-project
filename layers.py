import torch

from torch.nn.utils import weight_norm as wn
from torch import nn
from utils import concat_elu


class Nin(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Nin, self).__init__()
        self.lin_a = wn(nn.Linear(dim_in, dim_out))
        self.dim_out = dim_out

    def forward(self, x):
        """ A network in network layer (1x1 CONV) """
        x = x.permute(0, 2, 3, 1)
        shp = list(x.size())
        out = self.lin_a(x.contiguous().view(shp[0] * shp[1] * shp[2], shp[3]))
        shp[-1] = self.dim_out
        out = out.view(shp)
        return out.permute(0, 3, 1, 2)


class GatedResnet(nn.Module):
    """
    Args:
        skip_connection:    0 = no skip connection
                            1 = skip connection where skip input size === input size
                            2 = skip connection where skip input size === 2 * input size
    """
    def __init__(self, num_filters, conv_op, nonlinearity=concat_elu, skip_connection=0):
        super(GatedResnet, self).__init__()
        self.skip_connection = skip_connection
        self.nonlinearity = nonlinearity
        self.conv_input = conv_op(2 * num_filters, num_filters)  # because of concat_elu

        if skip_connection != 0:
            self.nin_skip = Nin(2 * skip_connection * num_filters, num_filters)

        # self.dropout = nn.Dropout2d(0.5)
        self.conv_out = conv_op(2 * num_filters, 2 * num_filters)

    def forward(self, og_x, a=None):
        x = self.conv_input(self.nonlinearity(og_x))
        if a is not None:
            x += self.nin_skip(self.nonlinearity(a))
        x = self.nonlinearity(x)
        # x = self.dropout(x)
        x = self.conv_out(x)
        a, b = torch.chunk(x, 2, dim=1)
        c3 = a * torch.sigmoid(b)
        return og_x + c3
