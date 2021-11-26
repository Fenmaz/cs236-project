import os
import sys
import argparse
import torch
import numpy as np
import shutil

from torch.optim import lr_scheduler, Adam
from torchvision import datasets, transforms, utils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import LMPCNN
from utils import discretized_mix_logistic_loss, discretized_mix_logistic_loss_1d, \
    sample_from_discretized_mix_logistic, sample_from_discretized_mix_logistic_1d
from masking import get_masks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-o', '--save_dir', type=str, default='models',
                        help='Location for parameter checkpoints and samples')
    parser.add_argument('-d', '--dataset', type=str,
                        default='cifar', help='Can be either cifar|mnist')
    parser.add_argument('-p', '--log_every', type=int, default=50,
                        help='How many iterations between logging losses?')
    parser.add_argument('-t', '--save_interval', type=int, default=10,
                        help='Every how many epochs to write checkpoint/samples?')
    parser.add_argument('-r', '--load_params', type=str, default=None,
                        help='Restore training from previous model checkpoint?')
    parser.add_argument('--overwrite', type=bool, default=False,
                        help='Overwrite previous runs with the same hyper parameters')
    # model
    parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                        help='Number of residual blocks per stage of the model')
    parser.add_argument('-n', '--nr_filters', type=int, default=160,
                        help='Number of filters to use across the model. Higher = larger model.')
    parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                        help='Number of logistic components in the mixture. Higher = more flexible model')
    parser.add_argument('-l', '--lr', type=float,
                        default=0.0002, help='Base learning rate')
    parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                        help='Learning rate decay, applied every step of the optimization')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Batch size during training per GPU')
    parser.add_argument('-x', '--max_epochs', type=int,
                        default=5000, help='How many epochs to run in total?')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed to use')
    args = parser.parse_args()
    return args


def setup(args):
    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_name = 'lmpcnn_nr-resnet{}_nr-filters{}'.format(args.nr_resnet, args.nr_filters)
    if os.path.exists(os.path.join('runs', model_name)) and args.overwrite:
        shutil.rmtree(os.path.join('runs', model_name))
    else:
        assert not os.path.exists(os.path.join('runs', model_name)), '{} already exists!'.format(model_name)
    writer = SummaryWriter(log_dir=os.path.join('runs', model_name))

    return writer, model_name


def data_loader(args, rescaling):
    kwargs = {'num_workers': 8, 'pin_memory': True, 'drop_last': True}

    ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])

    if 'mnist' in args.dataset:
        train_loader = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, download=True,
                                                                  train=True, transform=ds_transforms),
                                                   batch_size=args.batch_size,
                                                   shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, train=False,
                                                                 transform=ds_transforms), batch_size=args.batch_size,
                                                  shuffle=True, **kwargs)

        loss_op = discretized_mix_logistic_loss_1d

        def sample_op(_x):
            return sample_from_discretized_mix_logistic_1d(_x, args.nr_logistic_mix)

    elif 'cifar' in args.dataset:
        train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True,
                                                                    download=True, transform=ds_transforms),
                                                   batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False,
                                                                   transform=ds_transforms), batch_size=args.batch_size,
                                                  shuffle=True, **kwargs)

        loss_op = discretized_mix_logistic_loss

        def sample_op(_x):
            return sample_from_discretized_mix_logistic(_x, args.nr_logistic_mix)

    else:
        raise Exception('{} dataset not in {{mnist, cifar10}}'.format(args.dataset))

    return train_loader, test_loader, loss_op, sample_op


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parse_args()
    writer, model_name = setup(args)

    def rescaling(_x):
        return (_x - .5) * 2.

    def rescaling_inv(_x):
        return .5 * _x + .5

    train, test, loss_op, sample_op = data_loader(args, rescaling)

    obs = (1, 28, 28) if 'mnist' in args.dataset else (3, 32, 32)
    input_channels = obs[0]
    sample_batch = 25
    sample_dim = [sample_batch]
    sample_dim.extend(obs)

    model = LMPCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters,
                   input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix).to(device)
    idx = np.array([(x, y) for x in range(obs[1]) for y in range(obs[2])])

    if args.load_params:
        # load_part_of_model(model, args.load_params)
        model.load_state_dict(torch.load(args.load_params))
        print('model parameters loaded')

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

    writes = 0
    model.train()
    inner_bar = tqdm(train, leave=False, file=sys.stdout)
    for epoch in range(args.max_epochs):
        inner_bar.set_description('Epoch {}'.format(epoch))
        train_loss = 0.
        for batch_idx, (x, _) in enumerate(train):
            inner_bar.update(1)
            # Generate a random ordering & corresponding masks
            np.random.shuffle(idx)
            mask_init, mask_undilated, mask_dilated = get_masks(idx, obs[1], obs[2], device=device)

            x = x.to(device)
            output = model(x, mask_init, mask_undilated, mask_dilated)
            loss = loss_op(x, output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data.item()
            if (batch_idx + 1) % args.log_every == 0:
                deno = args.log_every * args.batch_size * np.prod(obs) * np.log(2.)
                writer.add_scalar('train/bpd', (train_loss / deno), writes)
                inner_bar.set_postfix(loss='{:.4f}'.format(train_loss / deno))
                train_loss = 0.
                writes += 1

        # decrease learning rate
        scheduler.step()

        with torch.no_grad():
            # torch.cuda.synchronize()
            model.eval()
            test_loss = 0.
            batch_idx = 0
            for batch_idx, (x, _) in enumerate(test):
                np.random.shuffle(idx)
                mask_init, mask_undilated, mask_dilated = get_masks(idx, obs[1], obs[2], device=device)

                x = x.to(device)
                output = model(x, mask_init, mask_undilated, mask_dilated)
                loss = loss_op(x, output)
                test_loss += loss.data.item()
                del loss, output

            deno = batch_idx * args.batch_size * np.prod(obs) * np.log(2.)
            writer.add_scalar('test/bpd', (test_loss / deno), writes)
            inner_bar.set_postfix(test_loss='{:.4f}'.format(test_loss / deno))

            if (epoch + 1) % args.save_interval == 0:
                torch.save(model.state_dict(), 'models/{}_{}.pth'.format(model_name, epoch))
                model.eval()
                sample = torch.zeros(sample_dim).to(device)
                for i in range(obs[1]):
                    for j in range(obs[2]):
                        out = model(sample, sample=True)
                        out_sample = sample_op(out)
                        sample[:, :, i, j] = out_sample.data[:, :, i, j]
                sample_t = rescaling_inv(sample)
                utils.save_image(sample_t, 'images/{}_{}.png'.format(model_name, epoch),
                                 nrow=5, padding=0)

        inner_bar.reset()


if __name__ == '__main__':
    main()
