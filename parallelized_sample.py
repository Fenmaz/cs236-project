import os

import numpy as np
import torch
from torchvision import transforms, utils
from tqdm import tqdm

from main import parse_args, data_loader
from masking import get_masks
from model import LMPCNN
from sample_mnist import sample_op
from utils import discretized_mix_logistic_log_probs, rescaling_inv, rescaling

SAMPLE_BATCH = 25
OBS = (1, 28, 28)


def compute_loss_steps(model, device, idx, train):
    model.eval()
    mask_init, mask_undilated, mask_dilated = get_masks(idx, OBS[1], OBS[2], device=device)

    average_nlp = torch.zeros(OBS, device=device)
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(tqdm(train)):
            x = x.to(device)
            output = model(x, mask_init, mask_undilated, mask_dilated)
            log_probs = discretized_mix_logistic_log_probs(x, output)
            negative_log_probs = -torch.logsumexp(log_probs, dim=len(log_probs.size()) - 1)
            average_nlp += negative_log_probs.mean(dim=0)
        average_nlp /= (batch_idx + 1)

    return average_nlp.squeeze().cpu().numpy()


def get_nelbo_matrix(kl_per_t):
    """Computes nelbo matrix so that nelbos[s, t] is going to contain the value logp(x_s | x_t)."""
    num_timesteps = len(kl_per_t)

    # Somewhat unintuitive code to create this mask, looking at a print is way
    # more intuitive. In the case of num_timesteps = 3, it would look like:
    # [0 1 2 3]
    # [0 0 1 2]
    # [0 0 0 1]
    # [0 0 0 0], i.e. to build nelbos[s, t] = -log(x_s | x_t)
    triu = np.triu(np.ones((num_timesteps, num_timesteps)))
    triu = np.cumsum(triu[::-1], axis=0)[::-1]

    # Compute nelbos[s, t] is going to contain the value logp(x_s | x_t). Only
    # considering entries where s > t.
    nelbos_ = kl_per_t[:, None] * triu
    nelbos = np.zeros((num_timesteps + 1, num_timesteps + 1))
    # Last row / first column are zero to match up with costs/dimensions
    # matrices.
    nelbos[:-1, 1:] = nelbos_

    return nelbos


def get_cost_and_dimension_matrices_np(kl_per_t):
    """Compute cost and assignment matrices in numpy."""
    num_timesteps = len(kl_per_t)

    # costs[k, t] is going to contain the cost to generate t steps but limited to
    # a policy with length k.
    costs = np.full(
        (num_timesteps + 1, num_timesteps + 1), np.inf, dtype=float)
    costs[0, 0] = 0

    # dimensions[k, t] is going to contain the optimal previous t for D[k - 1].
    dimensions = np.full(
        (num_timesteps + 1, num_timesteps + 1), -1, dtype=np.int32)

    # nelbos[s, t] is going to contain the value logp(x_s | x_t)
    nelbos = get_nelbo_matrix(kl_per_t)

    # Compute cost and assignment matrices.
    for k in range(1, num_timesteps + 1):
        # More efficient, we only have to consider costs <=k:
        bpds = costs[k - 1, :k + 1, None] + nelbos[:k + 1, :]
        dimensions[k] = np.argmin(bpds, axis=0)
        # Use argmin to get minimum to save time, equiv to calling np.amin.
        amin = bpds[dimensions[k], np.arange(num_timesteps + 1)]
        costs[k] = amin

        # # Easier to interpret but more expensive, leaving it here for clarity:
        # bpds = costs[k-1, :, None] + nelbos
        # dimensions[k] = np.argmin(bpds, axis=0)
        # # Use argmin to get minimum to save time, equiv to calling np.amin.
        # amin = bpds[dimensions[k], np.arange(num_timesteps+1)]
        # costs[k] = amin

    return costs, dimensions


def get_optimal_path_with_budget(budget, costs, dimensions):
    """Compute optimal path."""
    t = costs.shape[0] - 1
    path = []
    opt_cost = costs[budget, t]
    for k in reversed(range(1, budget + 1)):
        t = dimensions[k, t]
        path.append(t)

    # Path is reversed, reverse back.
    path = list(reversed(path))

    return path, opt_cost


def sample(model, device, idx, path, filename):
    mask_init, mask_undilated, mask_dilated = get_masks(idx, OBS[1], OBS[2], device=device)
    sample_dim = [SAMPLE_BATCH] + list(OBS)
    num_dim = len(idx)
    path.append(num_dim)

    with torch.no_grad():
        sample = torch.zeros(sample_dim, device=device)
        # inner_bar = tqdm(idx, leave=False)
        for it in range(len(path) - 1):
            # inner_bar.update(path[it + 1] - path[it])
            out = model(sample, mask_init, mask_undilated, mask_dilated, sample=True)
            out_sample = sample_op(out)
            for i, j in idx[path[it]: path[it + 1]]:
                sample[:, :, i, j] = out_sample[:, :, i, j]
        sample_t = rescaling_inv(sample)
        utils.save_image(sample_t, 'images/{}.png'.format(filename), nrow=5, padding=0)

    return sample_t


def eval_test(idx, path, model, device, test):
    with torch.no_grad():
        model.eval()
        test_loss = 0.
        batch_idx = 0
        for batch_idx, (x, _) in enumerate(tqdm(test)):
            mask_init, mask_undilated, mask_dilated = get_masks(idx, OBS[1], OBS[2], device=device)

            x = x.to(device)
            output = model(x, mask_init, mask_undilated, mask_dilated)
            log_probs = discretized_mix_logistic_log_probs(x, output)
            loss = -torch.logsumexp(log_probs, dim=len(log_probs.size()) - 1).sum()
            test_loss += loss.data.item()
            del loss, output

        deno = batch_idx * args.batch_size * np.prod(OBS) * np.log(2.)
        print("Test loss on parallelized path: {}".format((test_loss / deno)))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_args()

    ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])
    kwargs = {'num_workers': 8, 'pin_memory': True, 'drop_last': True}
    train_loader, test_loader, _, _ = data_loader(args)

    model = LMPCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters,
                   input_channels=OBS[0], nr_logistic_mix=args.nr_logistic_mix).to(device)

    if args.load_params:
        model.load_state_dict(torch.load(args.load_params, map_location=device))
        model_filename = os.path.basename(args.load_params)

        idx = [(i, j) for i in range(OBS[1]) for j in range(OBS[2])]

        loss = compute_loss_steps(model, device, idx, train_loader)  # (28, 28)
        loss_array = np.array([loss[index] for index in idx])
        costs, dimensions = get_cost_and_dimension_matrices_np(loss_array)
        path, opt_cost = get_optimal_path_with_budget(20, costs, dimensions)
        print("Optimal path is {}".format(path))
        print(opt_cost)
        sample(model, device, idx, path, "parallelized_sample")
