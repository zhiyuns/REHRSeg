import os
import torch
import scipy
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F

from .parse_image_file import (
    parse_image,
    inv_normalize,
    lr_axis_to_z,
    z_axis_to_lr_axis,
)

from .patch_ops import find_integer_p, calc_slices_to_crop
from .rotate import rotate_vol_2d
from .timer import timer_context
from .blur_kernel_ops import calc_extended_patch_size, parse_kernel

def apply_to_vol_smore(model, image, batch_size):
    result = []
    num_batches = (image.shape[0] + batch_size - 1) // batch_size  # 计算批次数量
    for st in range(0, image.shape[0], batch_size):
        en = st + batch_size
        batch = image[st:en]
        batch = batch.permute(0, 1, 3, 2)
        with torch.inference_mode():
            sr = model(batch).detach().cpu()
        result.append(sr)
    result = torch.cat(result, dim=0)
    return result

def inference_smore(model, sr_mode, in_fpath, ref_fpath, out_fpath, slice_thickness, target_thickness, device):
    # ===== LOAD AND PROCESS DATA =====

    image, slice_separation, lr_axis, _, header, affine, orig_min, orig_max = parse_image(
        in_fpath, slice_thickness, target_thickness
    )
    if len(image.shape) == 3:
        image = image[...,np.newaxis]
        lr_axis = 0
    if sr_mode == 'img':
        image = image[...,0:1]
    image = lr_axis_to_z(image, lr_axis)
    # pad the number of slices out so we achieve the correct final resolution
    n_slices_pad = find_integer_p(image.shape[2], slice_separation)
    n_slices_crop = calc_slices_to_crop(n_slices_pad, slice_separation)
    image = np.pad(image, ((0, 0), (0, 0), (0, n_slices_pad), (0, 0)), mode="reflect")
    image = torch.from_numpy(image)
    
    angles = [90]
    model_preds = []
    for i, angle in enumerate(angles):
        # Rotate in-plane. Image starts as (hr_axis, hr_axis, lr_axis)
        image_rot = rotate_vol_2d(image.to(device), angle)
        # Ensure the LR axis is s.t. (hr_axis, C, lr_axis, hr_axis)
        image_rot = image_rot.permute(0, 3, 2, 1)  # No need to unsqueeze(1) here
        # Run model
        rot_result = apply_to_vol_smore(model, image_rot, 1)
        # Return to (hr_axis, hr_axis, lr_axis)
        result = rot_result.permute(0, 3, 1, 2)
        model_preds.append(rotate_vol_2d(result, 0))

    # ===== FINALIZE =====
    final_out = torch.mean(torch.stack(model_preds), dim=0)
    final_out = final_out.detach().cpu().numpy().astype(np.float32)

    # Re-crop to target shape
    if n_slices_crop != 0:
        final_out = final_out[:, :, :-n_slices_crop]
    # Reorient to original orientation
    final_out = z_axis_to_lr_axis(final_out, lr_axis)
    final_out = final_out.transpose(0, 3, 2, 1)[:,:,:,::-1]

    ref_img = sitk.ReadImage(ref_fpath)
    spacing = ref_img.GetSpacing()
    origin = ref_img.GetOrigin()
    direction = ref_img.GetDirection()
    
    final_out_img = sitk.GetImageFromArray(final_out[0])
    final_out_img.SetSpacing((spacing[0], spacing[1], spacing[2]/slice_separation))
    final_out_img.SetOrigin(origin)
    final_out_img.SetDirection(direction)
    ref_img = sitk.ReadImage(ref_fpath)
    output_file = out_fpath.replace(".nii.gz", "_img.nii.gz")
    sitk.WriteImage(final_out_img, output_file)

    if 'seg' in sr_mode:
        final_out_seg = final_out[1]
        final_out_seg[final_out_seg>0] = 1
        final_out_seg[final_out_seg<0] = 0
        final_out_seg = sitk.GetImageFromArray(final_out_seg.astype('uint8'))
        final_out_seg.SetSpacing((spacing[0], spacing[1], spacing[2]/slice_separation))
        final_out_seg.SetOrigin(origin)
        final_out_seg.SetDirection(direction)
        output_file = out_fpath.replace(".nii.gz", "_seg.nii.gz")
        sitk.WriteImage(final_out_seg, output_file)

    return


def apply_to_vol_flavr(model, image, pred_out_idx=None):
    result = []
    
    # pad to ensure that it can be multiplied by 16
    ori_x = image.shape[2]
    ori_y = image.shape[3]
    if image.shape[2] % 16 != 0:
        pad_x = 16 - image.shape[2] % 16
        image = torch.cat([image, torch.zeros(image.shape[0], image.shape[1], pad_x, image.shape[3]).cuda()], dim=2)
    if image.shape[3] % 16 != 0:
        pad_y = 16 - image.shape[3] % 16
        image = torch.cat([image, torch.zeros(image.shape[0], image.shape[1], image.shape[2], pad_y).cuda()], dim=3)
    
    for st in range(0, image.shape[0]-1):
        if st == 0:
            batch = image[0:3]
            batch = torch.cat([torch.zeros(4-batch.shape[0], *batch.shape[1:]).cuda(), batch], dim=0)
        elif st == image.shape[0]-2:
            batch = image[st-1:]
            batch = torch.cat([batch, torch.zeros(4-batch.shape[0], *batch.shape[1:]).cuda()], dim=0)
        else:
            batch = image[st-1:st+3]
        batch = batch.permute(1, 0, 3, 2).unsqueeze(0)
        batch_input = batch.clone()

        with torch.inference_mode():
            sr = model(batch_input)
            if pred_out_idx is not None and isinstance(sr, tuple):
                sr = sr[pred_out_idx]
            sr = sr.detach().cpu()
        result.append(sr[:,:,:,:ori_y,:ori_x])
    result = torch.cat(result, dim=2).squeeze(0)
    result = result.permute(1,0,2,3)
    return result

def inference_flavr(model, sr_mode, in_fpath, ref_fpath, out_fpath, slice_thickness, target_thickness, device, enable_uncertainty):
    image, slice_separation, lr_axis, _, _, _, orig_min, orig_max = parse_image(
            in_fpath, slice_thickness, target_thickness
    )
    image = lr_axis_to_z(image, lr_axis)
    # pad the number of slices out so we achieve the correct final resolution
    n_slices_pad = find_integer_p(image.shape[2], slice_separation)
    n_slices_crop = calc_slices_to_crop(n_slices_pad, slice_separation)
    image = np.pad(image, ((0, 0), (0, 0), (0, n_slices_pad), (0, 0)), mode="reflect")
    image = torch.from_numpy(image)
    ref_img = sitk.ReadImage(ref_fpath)
    spacing = ref_img.GetSpacing()
    origin = ref_img.GetOrigin()
    direction = ref_img.GetDirection()
    if 'img' in sr_mode or 'seg' in sr_mode:
        if sr_mode == 'img':
            image = image[...,0:1]
        elif sr_mode == 'seg':
            image = image[...,1:]

        angles = [0]
        model_preds = []

        for i, angle in enumerate(angles):
            # Rotate in-plane. Image starts as (hr_axis, hr_axis, lr_axis)
            image_rot = rotate_vol_2d(image.to(device), angle)
            # Ensure the LR axis is s.t. (hr_axis, C, lr_axis, hr_axis)
            image_rot = image_rot.permute(0, 3, 2, 1)  # No need to unsqueeze(1) here

            # Run model
            rot_result = apply_to_vol_flavr(model, image_rot, 0)
            # Return to (hr_axis, hr_axis, lr_axis)
            result = rot_result.permute(0, 3, 1, 2)
            model_preds.append(rotate_vol_2d(result, -angle))

        # ===== FINALIZE =====
        final_out = torch.mean(torch.stack(model_preds), dim=0)
        final_out = final_out.detach().cpu().numpy().astype(np.float32)
        final_out = inv_normalize(final_out, orig_min, orig_max, a=0, b=1)

        # Re-crop to target shape
        if n_slices_crop != 0:
            final_out = final_out[:, :, :-n_slices_crop]
        # Reorient to original orientation
        final_out = z_axis_to_lr_axis(final_out, lr_axis)

        final_out_img = sitk.GetImageFromArray(final_out[0])
        final_out_img.SetSpacing((spacing[0], spacing[1], spacing[2]/slice_separation))
        final_out_img.SetOrigin(origin)
        final_out_img.SetDirection(direction)
        output_file = out_fpath.replace(".nii.gz", "_img.nii.gz")
        sitk.WriteImage(final_out_img, output_file)

        final_out_seg = final_out[1]
        final_out_seg[final_out_seg>0] = 1
        final_out_seg[final_out_seg<0] = 0
        final_out_seg = sitk.GetImageFromArray(final_out_seg.astype('uint8'))
        final_out_seg.SetSpacing((spacing[0], spacing[1], spacing[2]/slice_separation))
        final_out_seg.SetOrigin(origin)
        final_out_seg.SetDirection(direction)
        output_file = out_fpath.replace(".nii.gz", "_seg.nii.gz")
        sitk.WriteImage(final_out_seg, output_file)

    if enable_uncertainty:
        angles = [0]
        model_preds = []

        for i, angle in enumerate(angles):
            # Rotate in-plane. Image starts as (hr_axis, hr_axis, lr_axis)
            image_rot = rotate_vol_2d(image.to(device), angle)
            # Ensure the LR axis is s.t. (hr_axis, C, lr_axis, hr_axis)
            image_rot = image_rot.permute(0, 3, 2, 1)  # No need to unsqueeze(1) here

            # Run model
            rot_result = apply_to_vol_flavr(model, image_rot, 1)
            # Return to (hr_axis, hr_axis, lr_axis)
            result = rot_result.permute(0, 3, 1, 2)
            model_preds.append(rotate_vol_2d(result, -angle))

        # ===== FINALIZE =====
        final_out = torch.mean(torch.stack(model_preds), dim=0)
        final_out = final_out.detach().cpu().numpy().astype(np.float32)
        final_out = inv_normalize(final_out, orig_min, orig_max, a=0, b=1)

        # Re-crop to target shape
        if n_slices_crop != 0:
            final_out = final_out[:, :, :-n_slices_crop]
        # Reorient to original orientation
        final_out = z_axis_to_lr_axis(final_out, lr_axis)
        # Re-crop to target shape
        if n_slices_crop != 0:
            final_out = final_out[:, :, :-n_slices_crop]
        # Reorient to original orientation
        final_out = z_axis_to_lr_axis(final_out, lr_axis)
        final_out = final_out.transpose(1,2,0,3)

        spacing = ref_img.GetSpacing()
        origin = ref_img.GetOrigin()
        direction = ref_img.GetDirection()

        final_out_img = sitk.GetImageFromArray(final_out[0])
        final_out_img.SetSpacing((spacing[0], spacing[1], spacing[2]/slice_separation))
        final_out_img.SetOrigin(origin)
        final_out_img.SetDirection(direction)
        output_file = out_fpath.replace(".nii.gz", "_uncertainty.nii.gz")
        sitk.WriteImage(final_out_img, output_file)

def postprocess_smore(subject, slice_seperation=4, data_path=None, sr_path=None):
    if sr_path is not None:
        # use smore to interpolate the image
        sr_image, slice_separation, _,  blur_fwhm, _, _, _, _ = parse_image(
            os.path.join(sr_path, subject.replace('.nii.gz', '_img.nii.gz')), slice_seperation, 1.0
        )
        sr_label, slice_separation, _,  blur_fwhm, _, _, _, _ = parse_image(
            os.path.join(sr_path, subject.replace('.nii.gz', '_seg.nii.gz')), slice_seperation, 1.0
        )
        sr_image = sr_image[..., np.newaxis]
        sr_label = sr_label[..., np.newaxis]
        image = np.concatenate([sr_image, sr_label], axis=-1)
    else:
        # use traditional method to interpolate the image
        image, slice_separation, _,  blur_fwhm, _, _, _, _ = parse_image(
            os.path.join(data_path, subject), slice_seperation, 1.0
        )
        img_data = image[...,0]
        label_data = image[...,1]
        upsampled_img_data = scipy.ndimage.zoom(img_data, (1,1,slice_seperation), order=3)  # order=3 for cubic B-spline
        upsampled_label_data = scipy.ndimage.zoom(label_data, (1,1,slice_seperation), order=0)  # order=0 for nearest interpolation
        image = np.stack([upsampled_img_data, upsampled_label_data], axis=-1)
    
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

def zeroonenorm(data):
    data =  (data - np.min(data)) / (np.max(data) - np.min(data))
    data = (data * 255.0)
    return data

def postprocess_flavr(subject, slice_seperation=4, sr_path=None):
    image, slice_separation, _,  blur_fwhm, _, _, _, _ = parse_image(
        os.path.join(sr_path, subject.replace('.nii.gz', '_img.nii.gz')), slice_seperation, 1.0
    )
    image = zeroonenorm(image)
    label, slice_separation, _,  blur_fwhm, _, _, _, _ = parse_image(
        os.path.join(sr_path, subject.replace('.nii.gz', '_seg.nii.gz')), slice_seperation, 1.0
    )
    if os.path.exists(os.path.join(sr_path, subject.replace('_img', '_uncertainty'))):
        uncertainty, _, _,  _, _, _, _, _ = parse_image(
            os.path.join(sr_path, subject.replace('_img', '_uncertainty')), 4.0, 1.0
        )
        uncertainty = (zeroonenorm(uncertainty) * 255.0).astype('uint8')
    else:
        uncertainty = np.zeros_like(label)

    blur_kernel = parse_kernel(None, 'rf-pulse-slr', blur_fwhm)
    image_torch = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(1) # z, x, y
    image_torch_blurred = F.conv2d(image_torch, blur_kernel, padding="same").squeeze(1).numpy()
    image = image_torch_blurred.transpose(1, 2, 0)
    return image, label, uncertainty