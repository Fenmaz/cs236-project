import os
import numpy as np
import torch


def kernel_masks(generation_order_idx: np.ndarray, nrows, ncols, kernel_size=3,
                 dilation=1, mask_type='B', set_padding=0, observed_idx: np.ndarray = None) -> np.ndarray:
    """Generate kernel masks given a pixel generation order.

    Args:
        generation_order_idx: N x 2 array, order to generate pixels.
        nrows
        ncols
        kernel_size
        dilation
        mask_type: A or B
        set_padding
        observed_idx: M x 2 array, for coords in this list, will allow all locations to condition.
            Useful for inpainting tasks, where some context is observed and masking is only needed
            in the unobserved region.
    """
    assert kernel_size % 2 == 1, "Only odd sized kernels are implemented"
    half_k = int(kernel_size / 2)
    masks = np.zeros((len(generation_order_idx), kernel_size, kernel_size))

    locs_generated = set()
    if observed_idx is not None:
        # Can observe some context
        for r, c in observed_idx:
            locs_generated.add((r, c))

    # Set masks
    for i, (r, c) in enumerate(generation_order_idx):
        row_major_index = r * ncols + c
        for dr in range(-half_k, half_k + 1):
            for dc in range(-half_k, half_k + 1):
                if dr == 0 and dc == 0:
                    # skip center pixel of mask
                    continue

                loc = (r + dr * dilation, c + dc * dilation)
                if loc in locs_generated:
                    # The desired location has been generated,
                    # so we can condition on it
                    masks[row_major_index, half_k + dr, half_k + dc] = 1
                elif not (0 <= loc[0] < nrows and 0 <= loc[1] < ncols):
                    # Kernel location overlaps with padding
                    masks[row_major_index, half_k + dr, half_k + dc] = set_padding
        locs_generated.add((r, c))

    if mask_type == 'B':
        masks[:, half_k, half_k] = 1
    else:
        assert np.all(masks[:, half_k, half_k] == 0)

    return masks


def get_unfolded_masks(generation_order_idx, nrows, ncols, kernel_size=3, dilation=1, mask_type='B', observed_idx=None):
    assert mask_type in ['A', 'B']
    masks = kernel_masks(generation_order_idx, nrows, ncols, kernel_size, dilation, mask_type,
                         set_padding=0, observed_idx=observed_idx)
    masks = torch.tensor(masks, dtype=torch.float)
    masks_unf = masks.view(1, nrows * ncols, -1).transpose(1, 2)
    return masks_unf


def get_masks(generation_idx, nrows: int, ncols: int, kernel_size: int = 5, max_dilation: int = 2, observed_idx=None,
              device="cpu"):
    """Get and plot three masks: mask type A for first layer, mask type B for later layers,
    and mask type B with dilation. Masks are copied to GPU."""
    mask_init = get_unfolded_masks(generation_idx, nrows, ncols, kernel_size=kernel_size, dilation=1, mask_type='A',
                                   observed_idx=observed_idx).to(device=device, non_blocking=True)
    mask_undilated = get_unfolded_masks(generation_idx, nrows, ncols, kernel_size=kernel_size, dilation=1,
                                        mask_type='B',
                                        observed_idx=observed_idx).to(device=device, non_blocking=True)

    if max_dilation == 1:
        mask_dilated = mask_undilated
    else:
        mask_dilated = get_unfolded_masks(generation_idx, nrows, ncols, kernel_size=kernel_size, dilation=max_dilation,
                                          mask_type='B', observed_idx=observed_idx).to(device=device, non_blocking=True)

    return mask_init, mask_undilated, mask_dilated
