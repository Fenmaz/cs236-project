from layers import *
from utils import *


class PixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList([GatedResnet(nr_filters, down_shifted_conv2d,
                                                   resnet_nonlinearity, skip_connection=0)
                                       for _ in range(nr_resnet)])

        # stream from pixels above and to the left
        self.ul_stream = nn.ModuleList([GatedResnet(nr_filters, down_right_shifted_conv2d,
                                                    resnet_nonlinearity, skip_connection=1)
                                        for _ in range(nr_resnet)])

    def forward(self, u, ul):
        u_list, ul_list = [], []

        for i in range(self.nr_resnet):
            u = self.u_stream[i](u)
            ul = self.ul_stream[i](ul, a=u)
            u_list += [u]
            ul_list += [ul]

        return u_list, ul_list


class PixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList([GatedResnet(nr_filters, down_shifted_conv2d,
                                                   resnet_nonlinearity, skip_connection=1)
                                       for _ in range(nr_resnet)])

        # stream from pixels above and to the left
        self.ul_stream = nn.ModuleList([GatedResnet(nr_filters, down_right_shifted_conv2d,
                                                    resnet_nonlinearity, skip_connection=2)
                                        for _ in range(nr_resnet)])

    def forward(self, u, ul, u_list, ul_list):
        for i in range(self.nr_resnet):
            u = self.u_stream[i](u, a=u_list.pop())
            ul = self.ul_stream[i](ul, a=torch.cat((u, ul_list.pop()), 1))

        return u, ul


class PixelCNN(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10, input_channels=3):
        super(PixelCNN, self).__init__()
        self.resnet_nonlinearity = concat_elu
        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad = nn.ZeroPad2d((0, 0, 1, 0))

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([PixelCNNLayer_down(down_nr_resnet[i], nr_filters,
                                                             self.resnet_nonlinearity) for i in range(3)])

        self.up_layers = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters,
                                                         self.resnet_nonlinearity) for _ in range(3)])

        self.downsize_u_stream = nn.ModuleList([down_shifted_conv2d(nr_filters, nr_filters,
                                                                    stride=(2, 2)) for _ in range(2)])

        self.downsize_ul_stream = nn.ModuleList([down_right_shifted_conv2d(nr_filters,
                                                                           nr_filters, stride=(2, 2)) for _ in
                                                 range(2)])

        self.upsize_u_stream = nn.ModuleList([down_shifted_deconv2d(nr_filters, nr_filters,
                                                                    stride=(2, 2)) for _ in range(2)])

        self.upsize_ul_stream = nn.ModuleList([down_right_shifted_deconv2d(nr_filters,
                                                                           nr_filters, stride=(2, 2)) for _ in
                                               range(2)])

        self.u_init = down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2, 3),
                                          shift_output_down=True)

        self.ul_init = nn.ModuleList([down_shifted_conv2d(input_channels + 1, nr_filters,
                                                          filter_size=(1, 3), shift_output_down=True),
                                      down_right_shifted_conv2d(input_channels + 1, nr_filters,
                                                                filter_size=(2, 1), shift_output_right=True)])

        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None

    def forward(self, x, sample=False):
        # similar as done in the tf repo :
        if self.init_padding is None and not sample:
            padding_s = list(x.size())
            padding_s[1] = 1
            padding = torch.ones(padding_s, requires_grad=False)
            self.init_padding = padding.cuda() if x.is_cuda else padding

        if sample:
            padding_s = list(x.size())
            padding_s[1] = 1
            padding = torch.ones(padding_s, requires_grad=False)
            padding = padding.cuda() if x.is_cuda else padding
            x = torch.cat((x, padding), 1)

        # Upstream
        x = x if sample else torch.cat((x, self.init_padding), 1)
        u_list = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
        for i in range(3):
            # resnet block
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1])
            u_list += u_out
            ul_list += ul_out

            if i != 2:
                # downscale (only twice)
                u_list += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]

        # Downstream
        u = u_list.pop()
        ul = ul_list.pop()

        for i in range(3):
            # resnet block
            u, ul = self.down_layers[i](u, ul, u_list, ul_list)

            # upscale (only twice)
            if i != 2:
                u = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)

        x_out = self.nin_out(F.elu(ul))
        return x_out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ''' testing loss with tf version '''
    np.random.seed(1)
    x_t = torch.rand(15, 100, 32, 32, dtype=torch.float32).to(device)
    x_t = x_t * 3
    y_t = torch.rand(15, 3, 32, 32).to(device)
    y_t = y_t * 2 - 1
    loss = discretized_mix_logistic_loss(y_t, x_t)
    print('loss : %s' % loss.data.item())

    ''' testing model and deconv dimensions '''
    x = torch.FloatTensor(32, 3, 32, 32).uniform_(-1., 1.).to(device)
    ds = down_shifted_deconv2d(3, 40, stride=(2, 2))

    ''' testing loss compatibility '''
    model = PixelCNN(nr_resnet=3, nr_filters=100, input_channels=x.size(1))
    model = model.to(device)
    out = model(x)
    loss = discretized_mix_logistic_loss(x, out)
    print('loss : %s' % loss.data.item())
