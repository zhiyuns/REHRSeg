import numpy as np
import torch
import torch.nn.functional as F

def get_pads(target_dim, d):
    if target_dim <= d:
        return 0, 0
    p = (target_dim - d) // 2
    # if (p * 2 + d) % 2 != 0:
    #     return p, p + 1
    return p, target_dim - d - p


def target_pad(img, target_dims, mode="reflect"):
    pads = tuple(get_pads(t, d) for t, d in zip(target_dims, img.shape))
    if isinstance(img, torch.Tensor):
        img = img.numpy()
        return torch.Tensor(np.pad(img, pads, mode=mode)), pads
    else:
        return np.pad(img, pads, mode=mode), pads


def format_pads(pads):
    """Turn pad amounts into appropriate slices and handle 0 pads as None slices"""
    st = pads[0] if pads[0] != 0 else None
    en = -pads[1] if pads[1] != 0 else None
    return slice(st, en)


def crop(img, pads):
    crops = tuple(map(format_pads, pads))
    return img[crops]
