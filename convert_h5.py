# %%
import h5py
import torch
import numpy as np
import os
import scipy
from tqdm import tqdm

ori_path = r'./datasets/Brain_datasets/trainnew'
sr_path = r'./checkpoints/Brain_datasets/smore_2channel/eval'
out_path = r'./datasets/Brain_datasets/trainnew_h5'
all_subject = os.listdir(ori_path)

# %%
import torch.nn.functional as F
from utils.parse_image_file import parse_image
from utils.blur_kernel_ops import calc_extended_patch_size, parse_kernel
from utils.pad import target_pad

# %%
def preprocess(subject):
    image, slice_separation, _,  blur_fwhm, _, _, _, _ = parse_image(
        os.path.join(ori_path, subject), 4.0, 1.0
    )
    sr_image, slice_separation, _,  blur_fwhm, _, _, _, _ = parse_image(
        os.path.join(sr_path, subject.replace('.nii.gz', '_img.nii.gz')), 4.0, 1.0
    )
    sr_label, slice_separation, _,  blur_fwhm, _, _, _, _ = parse_image(
        os.path.join(sr_path, subject.replace('.nii.gz', '_seg.nii.gz')), 4.0, 1.0
    )
    image = image.squeeze() # shape (x, y, z, 2)
    sr_image = sr_image[..., np.newaxis]
    sr_label = sr_label[..., np.newaxis]
    if len(image.shape) == 3:
        image = image[..., np.newaxis]
    
    # upsample the image using b-spline interpolation
    '''
    img_data = image[...,0]
    label_data = image[...,1]
    upsampled_img_data = scipy.ndimage.zoom(img_data, (1,1,2), order=3)  # order=3 for cubic B-spline
    upsampled_label_data = scipy.ndimage.zoom(label_data, (1,1,2), order=0)  # order=0 for nearest interpolation
    image = np.stack([upsampled_img_data, upsampled_label_data], axis=-1)
    '''
    image = np.concatenate([sr_image, sr_label], axis=-1)
    
    blur_kernel = parse_kernel(None, 'rf-pulse-slr', blur_fwhm)
    img_hr = image[...,:1]
    label_hr = image[...,1:].astype('uint8')
    image_x = torch.from_numpy(image.transpose(2, 3, 0, 1)) # z, channel, x, y
    image_x_rgb = image_x[:, 0:1, ...]
    image_x_rgb = F.conv2d(image_x_rgb, blur_kernel, padding="same").numpy()

    image_y = torch.from_numpy(image.transpose(2, 3, 1, 0)) # z, channel, y, x
    image_y_rgb = image_y[:, 0:1, ...]
    image_y_rgb = F.conv2d(image_y_rgb, blur_kernel, padding="same").numpy()
    return img_hr, label_hr, image_x_rgb, image_y_rgb

# %%
for subject in tqdm(all_subject):
    img_hr, label_hr, image_x_rgb, image_y_rgb = preprocess(subject)
    with h5py.File(os.path.join(out_path, subject + '.h5'), 'w') as f:
        f.create_dataset('img_hr', data=img_hr)
        f.create_dataset('label_hr', data=label_hr)
        f.create_dataset('image_x_rgb', data=image_x_rgb)
        f.create_dataset('image_y_rgb', data=image_y_rgb)



