import torch
import random
import torchvision.transforms.functional as F

def augment_3d_image(image):
    """
    As augmentation, we hold all combinations of horizontal and vertical flips
    as well as transposition for the in-plane of a 3D image.

    This function is written by GPT-4 Code Interpreter and edited by a human.
    """
    if random.random() < 0.5:
        image = F.hflip(image)
    if random.random() < 0.5:
        image = F.vflip(image)

    return image
