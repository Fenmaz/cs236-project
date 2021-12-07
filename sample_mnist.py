import os

import numpy as np
import torch
from torchinfo import summary
from torchvision import utils
from tqdm import tqdm

from main import parse_args
from masking import get_masks
from model import LMPCNN
from utils import sample_from_discretized_mix_logistic_1d, rescaling_inv

SAMPLE_BATCH = 25
OBS = (1, 28, 28)


def sample_op(_x):
    return sample_from_discretized_mix_logistic_1d(_x, 10)


def sample_random(model, filename, device):
    idx = np.array([(x, y) for x in range(OBS[1]) for y in range(OBS[2])])
    np.random.shuffle(idx)
    mask_init, mask_undilated, mask_dilated = get_masks(idx, OBS[1], OBS[2], device=device)
    sample_dim = [SAMPLE_BATCH] + list(OBS)

    with torch.no_grad():
        sample = torch.zeros(sample_dim).to(device)
        for i, j in tqdm(idx):
            out = model(sample, mask_init, mask_undilated, mask_dilated, sample=True)
            out_sample = sample_op(out)
            sample[:, :, i, j] = out_sample.data[:, :, i, j]
        sample_t = rescaling_inv(sample)
        utils.save_image(sample_t, 'images/{}.png'.format(filename), nrow=5, padding=0)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_args()
    model = LMPCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters,
                   input_channels=OBS[0], nr_logistic_mix=args.nr_logistic_mix).to(device)
    summary(model, [[args.batch_size] + list(OBS)] + [[OBS[0], 5 * 5, OBS[1] * OBS[2]]] * 3)

    if args.load_params:
        model.load_state_dict(torch.load(args.load_params, map_location=device))
        model_filename = os.path.basename(args.load_params)
        sample_random(model, model_filename, device)
