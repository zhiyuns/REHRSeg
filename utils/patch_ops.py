import numpy as np
from scipy.ndimage import gaussian_filter
from math import floor


def projected_size(n_slices, p, scale):
    """
    The projected number of slices after initially padding `n_slices`
    by `p`. We would like to choose `p` to match the results of `ideal_slice()`.
    """
    scale_tilde = scale / floor(scale)
    return round((n_slices + p) * scale_tilde) * floor(scale) - round(p * scale)


def calc_slices_to_crop(p, scale):
    return round(p * scale)


def ideal_size(n_slices, scale):
    """
    The correct number of slices according to `resize`, which uses
    a `round` operation to get an integer number of slices.
    """
    return round(n_slices * scale)


def find_integer_p(n_slices, s):
    """
    The goal here is to, at test time, pad out the number of slices, then
    run the RCAN model, then crop off all the extra slices we got from the
    initial padding. This function finds the padding which achieves this.
    """
    p = 0  # Start testing from p = 0
    max_iter = 1000  # Maximum number of iterations to prevent infinite loop
    iter_count = 0  # Counter for the number of iterations

    while projected_size(n_slices, p, s) != ideal_size(n_slices, s) and iter_count < max_iter:
        p += 1
        iter_count += 1

    # If solution is found within max_iter iterations, return p
    if projected_size(n_slices, p, s) == ideal_size(n_slices, s):
        return p
    # If no solution is found within max_iter iterations, it is unachievable
    # and we just don't do any padding.
    return 0


def get_patch(img_rot, patch_center, patch_size, return_idx=False):
    """
    img_rot: np.array, the HR in-plane image at a single rotation
    patch_center: tuple of ints, center position of the patch
    patch_size: tuple of ints, the patch size in 3D. For 2D patches, supply (X, Y, 1).
    """

    # Get random rotation and center
    sts = [c - p // 2 if p != 1 else c for c, p in zip(patch_center, patch_size)]
    ens = [st + p for st, p in zip(sts, patch_size)]
    idx = tuple(slice(st, en) for st, en in zip(sts, ens))

    if return_idx:
        return idx

    return img_rot[idx].squeeze()


def get_random_centers(imgs_rot, patch_size, n_patches, weighted=True):
    rot_choices = np.random.randint(0, len(imgs_rot), size=n_patches)
    centers = []

    for i, img_rot in enumerate(imgs_rot):
        n_choices = int(np.sum(rot_choices == i))

        if weighted:
            smooth = gaussian_filter(img_rot, 1.0)
            grads = np.gradient(smooth)
            grad_mag = np.sum([np.sqrt(np.abs(grad)) for grad in grads], axis=0)

            # Set probability to zero at edges
            for p, axis in zip(patch_size, range(grad_mag.ndim)):
                if p > 1:
                    grad_mag = np.swapaxes(grad_mag, 0, axis)
                    grad_mag[: p // 2 + 1] = 0.0
                    grad_mag[-p // 2 - 1 :] = 0.0
                    grad_mag = np.swapaxes(grad_mag, axis, 0)

            # Normalize gradient magnitude to create probabilities            
            grad_probs = grad_mag / grad_mag.sum()
            grad_probs = [
                grad_probs.sum(axis=tuple(k for k in range(grad_probs.ndim) if k != axis))
                for axis in range(len(grad_probs.shape))
            ]
            # Re-normalize per axis to ensure probabilities sum to 1
            for axis in range(len(grad_probs)):
                grad_probs[axis] = grad_probs[axis] / grad_probs[axis].sum()
            
        else:
            grad_probs = [None for _ in img_rot.shape]


        # Generate random patch centers for each dimension
        random_indices = [
            np.random.choice(
                np.arange(0, img_dim),
                size=n_choices,
                p=grad_probs[axis],
            )
            for axis, img_dim in enumerate(img_rot.shape)
        ]
        # Combine random indices to form multi-dimensional patch centers
        centers.extend((i, tuple(coord)) for coord in zip(*random_indices))
    np.random.shuffle(centers)
    return centers
