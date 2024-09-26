import torch
from torch import nn
import SimpleITK as sitk
import numpy as np
import itertools

from acvl_utils.cropping_and_padding.padding import pad_nd_image

from nnunetv2.training.loss.dice import SoftDiceLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from nnunetv2.inference.sliding_window_prediction import compute_gaussian

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

from nnunetv2.training.data_augmentation.custom_transforms.cascade_transforms import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, RemoveRandomConnectedComponentFromOneHotEncodingTransform
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import \
    DownsampleSegForDSTransform2
from nnunetv2.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import \
    LimitedLenWrapper
from nnunetv2.training.data_augmentation.custom_transforms.masking import MaskTransform
from nnunetv2.training.data_augmentation.custom_transforms.region_based_training import \
    ConvertSegmentationToRegionsTransform
from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert2DTo3DTransform, \
    Convert3DTo2DTransform
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor

from batchgenerators.augmentations.utils import create_zero_centered_coordinate_mesh, elastic_deform_coordinates, \
    interpolate_img, \
    rotate_coords_2d, rotate_coords_3d, scale_coords, resize_segmentation, resize_multichannel_image, \
    elastic_deform_coordinates_2
from batchgenerators.augmentations.crop_and_pad_augmentations import random_crop as random_crop_aug
from batchgenerators.augmentations.crop_and_pad_augmentations import center_crop as center_crop_aug

import numpy as np
from multiprocessing import Pool

def read_image(image_fname):
    spacings_for_nnunet = []
    itk_image = sitk.ReadImage(image_fname)
    spacing = itk_image.GetSpacing()
    origin = itk_image.GetOrigin()
    direction = itk_image.GetDirection()
    npy_image = sitk.GetArrayFromImage(itk_image)
    if npy_image.ndim == 3:
        # 3d, as in original nnunet
        npy_image = npy_image[None]
        spacings_for_nnunet.append(list(spacing)[::-1])
    else:
        raise RuntimeError(f"Unexpected number of dimensions: {npy_image.ndim} in file {f}")

    spacings_for_nnunet[-1] = list(np.abs(spacings_for_nnunet[-1]))

    dict = {
        'sitk_stuff': {
            # this saves the sitk geometry information. This part is NOT used by nnU-Net!
            'spacing': spacing,
            'origin': origin,
            'direction': direction
        },
        # the spacing is inverted with [::-1] because sitk returns the spacing in the wrong order lol. Image arrays
        # are returned x,y,z but spacing is returned z,y,x. Duh.
        'spacing': spacings_for_nnunet[0]
    }
    return npy_image.astype(np.float32), dict

def _percentile_norm(input_tensor, reference_tensor=None, p_min=0.5, p_max=99.5, strictlyPositive=True):
        """Normalizes a tensor based on percentiles. Clips values below and above the percentile.
        Percentiles for normalization can come from another tensor.

        Args:
            input_tensor (torch.Tensor): Tensor to be normalized based on the data from the reference_tensor.
                If reference_tensor is None, the percentiles from this tensor will be used.
            reference_tensor (torch.Tensor, optional): The tensor used for obtaining the percentiles.
            p_min (float, optional): Lower end percentile. Defaults to 0.5.
            p_max (float, optional): Upper end percentile. Defaults to 99.5.
            strictlyPositive (bool, optional): Ensures that really all values are above 0 before normalization. Defaults to True.

        Returns:
            torch.Tensor: The input_tensor normalized based on the percentiles of the reference tensor.
        """
        if(reference_tensor == None):
            reference_tensor = input_tensor
        v_min, v_max = np.percentile(reference_tensor, [p_min,p_max]) #get p_min percentile and p_max percentile

        if( v_min < 0 and strictlyPositive): #set lower bound to be 0 if it would be below
            v_min = 0
        output_tensor = np.clip(input_tensor,v_min,v_max) #clip values to percentiles from reference_tensor
        output_tensor = (output_tensor - v_min)/(v_max-v_min) #normalizes values to [0;1]

        return output_tensor

def percentile_normalization(image, p_min=0.5, p_max=99.5, strictlyPositive=True):
    # tensor or numpy array?
    out_imgs = []
    if isinstance(image, torch.Tensor):
        for i in range(image.shape[0]):
            img = image[i:i+1,0,...].cpu().numpy()
            img = _percentile_norm(img, p_min=p_min, p_max=p_max, strictlyPositive=strictlyPositive)
            out_imgs.append(torch.from_numpy(img))
        out_imgs = torch.stack(out_imgs, dim=0)
        image = out_imgs
    else:
        image = image.astype(np.float32, copy=False)
        img = _percentile_norm(image, p_min=p_min, p_max=p_max, strictlyPositive=strictlyPositive)
        image = img
    return image

def zeroone_normalization(image):
    # tensor or numpy array?
    out_imgs = []
    if isinstance(image, torch.Tensor):
        for i in range(image.shape[0]):
            img = image[i:i+1,0,...]
            min = img.min()
            max = img.max()
            img -= min
            img /= (max - min)
            out_imgs.append(img)
        out_imgs = torch.stack(out_imgs, dim=0)
        image = out_imgs
    else:
        image = image.astype(np.float32, copy=False)
        min = image.min()
        max = image.max()
        image -= min
        image /= (max - min)
    return image

def zscore_normalization(image):
    # tensor or numpy array?
    out_imgs = []
    if isinstance(image, torch.Tensor):
        for i in range(image.shape[0]):
            img = image[i:i+1,0,...]
            mean = img.mean()
            std = img.std()
            img -= mean
            img /= (max(std, 1e-8))
            out_imgs.append(img)
        out_imgs = torch.stack(out_imgs, dim=0)
        image = out_imgs
    else:
        image = image.astype(np.float32, copy=False)
        mean = image.mean()
        std = image.std()
        image -= mean
        image /= (max(std, 1e-8))
    return image

def preprocess_image(image_file, apply_norm=True):
    data, properties = read_image(image_file)
    data = np.copy(data)

    # apply transpose_forward, this also needs to be applied to the spacing!
    data = data.transpose([0, *[i + 1 for i in [0,1,2]]])

    # normalize
    # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
    # longer fitting the images perfectly!
    if apply_norm:
        data = zscore_normalization(data)

    # print('current shape', data.shape[1:], 'current_spacing', original_spacing,
    #       '\ntarget shape', new_shape, 'target_spacing', target_spacing)
    data = torch.from_numpy(data).contiguous().float()
    return data, properties

def compute_steps_for_sliding_window(image_size, tile_size, tile_step_size):
    assert [i >= j for i, j in zip(image_size, tile_size)], "image size must be as large or larger than patch_size"
    assert 0 < tile_step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
    target_step_sizes_in_voxels = [i * tile_step_size for i in tile_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)]

    steps = []
    for dim in range(len(tile_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - tile_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)

    return steps

def _internal_maybe_mirror_and_predict(model, x, out_idx=None, deep_supervision=True, save=False) -> torch.Tensor:
        mirror_axes = (0,1,2)

        prediction = model(x)[out_idx] if out_idx is not None else model(x)
        if out_idx == 0 and deep_supervision:
            prediction = prediction[0]
        
        if mirror_axes is not None:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations([m + 2 for m in mirror_axes], i + 1)
            ]
            for axes in axes_combinations:
                if out_idx is None:
                    prediction += torch.flip(model(torch.flip(x, (*axes,))), (*axes,))
                else:
                    if out_idx == 0 and deep_supervision:
                        prediction += torch.flip(model(torch.flip(x, (*axes,)))[out_idx][0], (*axes,))
                    else:
                        prediction += torch.flip(model(torch.flip(x, (*axes,)))[out_idx], (*axes,))

            prediction /= (len(axes_combinations) + 1)
        
        return prediction

def _internal_get_sliding_window_slicers(image_size, patch_size=[14, 320, 384], tile_step_size=0.5):
        slicers = []
        steps = compute_steps_for_sliding_window(image_size, patch_size, tile_step_size)
        for sx in steps[0]:
            for sy in steps[1]:
                for sz in steps[2]:
                    slicers.append(
                        tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                                zip((sx, sy, sz), patch_size)]]))
        return slicers

def _internal_predict_sliding_window_return_logits(data: torch.Tensor,
                                                   slicers,
                                                   network,
                                                   do_on_device=True,
                                                   out_idx=None,
                                                   slice_seperation=1,
                                                   patch_size=[14, 320, 384],
                                                   use_gaussian=False,
                                                   deep_supervision=True):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = torch.device('cuda') if do_on_device else torch.device('cpu')

        # move data to device
        data = data.to(results_device)

        # preallocate arrays
        predicted_logits = torch.zeros((2, data.shape[1]*slice_seperation, data.shape[2], data.shape[3]),
                                        dtype=torch.half,
                                        device=results_device)
        n_predictions = torch.zeros((data.shape[1]*slice_seperation, data.shape[2], data.shape[3]), dtype=torch.half, device=results_device)
        if use_gaussian:
            gaussian = compute_gaussian(tuple(patch_size), sigma_scale=1. / 8,
                                                value_scaling_factor=10,
                                                device=results_device)
        else:
            gaussian = 1

        for i, sl in enumerate(slicers):
            workon = data[sl][None]
            workon = workon.to(results_device, non_blocking=False)

            prediction = _internal_maybe_mirror_and_predict(network, workon, out_idx, deep_supervision, i==len(slicers)-1)
            prediction = prediction[0].to(results_device)
            modified_sl = [slice(None)] + [slice(sl[i].start * slice_seperation, sl[i].stop * slice_seperation) if i==1 else slice(sl[i].start, sl[i].stop) for i in range(1, len(sl))]
            modified_sl = tuple(modified_sl)
            predicted_logits[modified_sl] += prediction * gaussian
            n_predictions[modified_sl[1:]] += gaussian
        
        predicted_logits /= n_predictions
        # check for infs
        if torch.any(torch.isinf(predicted_logits)):
            raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                                'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                                'predicted_logits to fp32')
        if use_gaussian:
            compute_gaussian.cache_clear()
        
        return predicted_logits

class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def forward(self, input, target, uncertainty=None):
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        loss = super().forward(input, target.long())
        if uncertainty is not None:
            loss = loss * uncertainty
        loss = loss.mean()
        return loss

class DC_and_weighted_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None, 
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_weighted_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, uncertainty=None):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0], uncertainty) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result
    
def _build_loss(enable_deep_supervision=False, weight_dice=1):
    loss = DC_and_weighted_CE_loss({'batch_dice': False,
                            'smooth': 1e-5, 'do_bg': False, 'ddp': False}, {'reduction': 'none'}, weight_ce=1, weight_dice=weight_dice,
                            ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss)

    # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
    # this gives higher resolution outputs more weight in the loss

    if enable_deep_supervision:
        deep_supervision_scales = [[1.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.25, 0.25], [0.5, 0.125, 0.125], [0.25, 0.0625, 0.0625], [0.25, 0.03125, 0.03125]]
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
    return loss

def process_channel(params):
    data, coords, order_data, border_mode_data, border_cval_data, is_seg = tuple(params)
    return interpolate_img(data, coords, order_data, border_mode_data, cval=border_cval_data, is_seg=is_seg)

def augment_spatial(data, seg_list, patch_size, patch_center_dist_from_border=30,
                    do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                    do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                    do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                    border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, p_el_per_sample=1,
                    p_scale_per_sample=1, p_rot_per_sample=1, independent_scale_for_each_axis=False,
                    p_rot_per_axis: float = 1, p_independent_scale_per_axis: int = 1, multiprocess=False, enable_uncertainty=False,):
    dim = len(patch_size)
    seg_result = None
    if seg_list is not None:
        if dim == 2:
            seg_result = [np.zeros((x.shape[0], x.shape[1], patch_size[0], patch_size[1]), dtype=np.float32) for x in seg_list]
        else:
            seg_result = [np.zeros((x.shape[0], x.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                                  dtype=np.float32) for x in seg_list]

    if dim == 2:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
    else:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                               dtype=np.float32)

    if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
        patch_center_dist_from_border = dim * [patch_center_dist_from_border]

    for sample_id in range(data.shape[0]):
        coords = create_zero_centered_coordinate_mesh(patch_size)

        if do_elastic_deform and np.random.uniform() < p_el_per_sample:
            a = np.random.uniform(alpha[0], alpha[1])
            s = np.random.uniform(sigma[0], sigma[1])
            coords = elastic_deform_coordinates(coords, a, s)

        if do_rotation and np.random.uniform() < p_rot_per_sample:

            if np.random.uniform() <= p_rot_per_axis:
                a_x = np.random.uniform(angle_x[0], angle_x[1])
            else:
                a_x = 0

            if dim == 3:
                if np.random.uniform() <= p_rot_per_axis:
                    a_y = np.random.uniform(angle_y[0], angle_y[1])
                else:
                    a_y = 0

                if np.random.uniform() <= p_rot_per_axis:
                    a_z = np.random.uniform(angle_z[0], angle_z[1])
                else:
                    a_z = 0

                coords = rotate_coords_3d(coords, a_x, a_y, a_z)
            else:
                coords = rotate_coords_2d(coords, a_x)

        if do_scale and np.random.uniform() < p_scale_per_sample:
            if independent_scale_for_each_axis and np.random.uniform() < p_independent_scale_per_axis:
                sc = []
                for _ in range(dim):
                    if np.random.random() < 0.5 and scale[0] < 1:
                        sc.append(np.random.uniform(scale[0], 1))
                    else:
                        sc.append(np.random.uniform(max(scale[0], 1), scale[1]))
            else:
                if np.random.random() < 0.5 and scale[0] < 1:
                    sc = np.random.uniform(scale[0], 1)
                else:
                    sc = np.random.uniform(max(scale[0], 1), scale[1])

            coords = scale_coords(coords, sc)

        # now find a nice center location 
        for d in range(dim):
            if random_crop:
                ctr = np.random.uniform(patch_center_dist_from_border[d],
                                        data.shape[d + 2] - patch_center_dist_from_border[d])
            else:
                ctr = data.shape[d + 2] / 2. - 0.5
            coords[d] += ctr
        
        if not multiprocess:
            for channel_id in range(data.shape[1]):
                data_result[sample_id, channel_id] = interpolate_img(data[sample_id, channel_id], coords, order_data,
                                                                        border_mode_data, cval=border_cval_data)
            if seg_list is not None:
                for i, seg in enumerate(seg_list):
                    for channel_id in range(seg.shape[1]):
                        if i == len(seg_list) - 1 and enable_uncertainty:
                            seg_result[i][sample_id, channel_id] = interpolate_img(seg[sample_id, channel_id], coords, order_data,
                                                                                    border_mode_data, cval=border_cval_data,
                                                                                    is_seg=False)
                        else:
                            seg_result[i][sample_id, channel_id] = interpolate_img(seg[sample_id, channel_id], coords, order_seg,
                                                                                    border_mode_seg, cval=border_cval_seg,
                                                                                    is_seg=True)
        else:

            # Assuming 'data' is your input data array
            num_channels = data.shape[1]
            data_result = np.empty_like(data)

            # Create a pool of worker processes
            with Pool() as pool:
                channel_ids = range(num_channels)
                data = [data[sample_id, channel_id] for channel_id in channel_ids]
                params = [(data, coords, order_data, border_mode_data, border_cval_data, False) for data in data]
                results = pool.map(process_channel, params)

            # Assign the results to the 'data_result' array
            for channel_id, result in zip(channel_ids, results):
                data_result[sample_id, channel_id] = result

            if seg_list is not None:
                for i, seg in enumerate(seg_list):
                    num_channels = seg.shape[1]
                    seg_result[i] = np.empty_like(seg)

                    # Create a pool of worker processes
                    with Pool() as pool:
                        channel_ids = range(num_channels)
                        seg = [seg[sample_id, channel_id] for channel_id in channel_ids]
                        if i == len(seg_list) - 1 and enable_uncertainty:
                            params = [(seg, coords, order_data, border_mode_data, border_cval_data, False) for seg in seg]
                        else:
                            params = [(seg, coords, order_seg, border_mode_seg, border_cval_seg, True) for seg in seg]
                        results = pool.map(process_channel, params)

                    # Assign the results to the 'data_result' array
                    for channel_id, result in zip(channel_ids, results):
                        seg_result[i][sample_id, channel_id] = result

    return data_result, seg_result

class MySpatialTransform(AbstractTransform):
    """The ultimate spatial transform generator. Rotation, deformation, scaling, cropping: It has all you ever dreamed
    of. Computational time scales only with patch_size, not with input patch size or type of augmentations used.
    Internally, this transform will use a coordinate grid of shape patch_size to which the transformations are
    applied (very fast). Interpolation on the image data will only be done at the very end

    Args:
        patch_size (tuple/list/ndarray of int): Output patch size

        patch_center_dist_from_border (tuple/list/ndarray of int, or int): How far should the center pixel of the
        extracted patch be from the image border? Recommended to use patch_size//2.
        This only applies when random_crop=True

        do_elastic_deform (bool): Whether or not to apply elastic deformation

        alpha (tuple of float): magnitude of the elastic deformation; randomly sampled from interval

        sigma (tuple of float): scale of the elastic deformation (small = local, large = global); randomly sampled
        from interval

        do_rotation (bool): Whether or not to apply rotation

        angle_x, angle_y, angle_z (tuple of float): angle in rad; randomly sampled from interval. Always double check
        whether axes are correct!

        do_scale (bool): Whether or not to apply scaling

        scale (tuple of float): scale range ; scale is randomly sampled from interval

        border_mode_data: How to treat border pixels in data? see scipy.ndimage.map_coordinates

        border_cval_data: If border_mode_data=constant, what value to use?

        order_data: Order of interpolation for data. see scipy.ndimage.map_coordinates

        border_mode_seg: How to treat border pixels in seg? see scipy.ndimage.map_coordinates

        border_cval_seg: If border_mode_seg=constant, what value to use?

        order_seg: Order of interpolation for seg. see scipy.ndimage.map_coordinates. Strongly recommended to use 0!
        If !=0 then you will have to round to int and also beware of interpolation artifacts if you have more then
        labels 0 and 1. (for example if you have [0, 0, 0, 2, 2, 1, 0] the neighboring [0, 0, 2] bay result in [0, 1, 2])

        random_crop: True: do a random crop of size patch_size and minimal distance to border of
        patch_center_dist_from_border. False: do a center crop of size patch_size

        independent_scale_for_each_axis: If True, a scale factor will be chosen independently for each axis.
    """

    def __init__(self, patch_size, patch_center_dist_from_border=30,
                 do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                 do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                 do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                 border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, data_key="data",
                 label_key=["seg",], p_el_per_sample=1, p_scale_per_sample=1, p_rot_per_sample=1,
                 independent_scale_for_each_axis=False, p_rot_per_axis:float=1, p_independent_scale_per_axis: int=1, enable_uncertainty: bool=False):
        self.independent_scale_for_each_axis = independent_scale_for_each_axis
        self.p_rot_per_sample = p_rot_per_sample
        self.p_scale_per_sample = p_scale_per_sample
        self.p_el_per_sample = p_el_per_sample
        self.data_key = data_key
        self.label_key = label_key
        self.patch_size = patch_size
        self.patch_center_dist_from_border = patch_center_dist_from_border
        self.do_elastic_deform = do_elastic_deform
        self.alpha = alpha
        self.sigma = sigma
        self.do_rotation = do_rotation
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.do_scale = do_scale
        self.scale = scale
        self.border_mode_data = border_mode_data
        self.border_cval_data = border_cval_data
        self.order_data = order_data
        self.border_mode_seg = border_mode_seg
        self.border_cval_seg = border_cval_seg
        self.order_seg = order_seg
        self.random_crop = random_crop
        self.p_rot_per_axis = p_rot_per_axis
        self.p_independent_scale_per_axis = p_independent_scale_per_axis
        self.enable_uncertainty = enable_uncertainty

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg_list = [data_dict.get(x) for x in self.label_key]

        if self.patch_size is None:
            if len(data.shape) == 4:
                patch_size = (data.shape[2], data.shape[3])
            elif len(data.shape) == 5:
                patch_size = (data.shape[2], data.shape[3], data.shape[4])
            else:
                raise ValueError("only support 2D/3D batch data.")
        else:
            patch_size = self.patch_size

        ret_val = augment_spatial(data, seg_list, patch_size=patch_size,
                                  patch_center_dist_from_border=self.patch_center_dist_from_border,
                                  do_elastic_deform=self.do_elastic_deform, alpha=self.alpha, sigma=self.sigma,
                                  do_rotation=self.do_rotation, angle_x=self.angle_x, angle_y=self.angle_y,
                                  angle_z=self.angle_z, do_scale=self.do_scale, scale=self.scale,
                                  border_mode_data=self.border_mode_data,
                                  border_cval_data=self.border_cval_data, order_data=self.order_data,
                                  border_mode_seg=self.border_mode_seg, border_cval_seg=self.border_cval_seg,
                                  order_seg=self.order_seg, random_crop=self.random_crop,
                                  p_el_per_sample=self.p_el_per_sample, p_scale_per_sample=self.p_scale_per_sample,
                                  p_rot_per_sample=self.p_rot_per_sample,
                                  independent_scale_for_each_axis=self.independent_scale_for_each_axis,
                                  p_rot_per_axis=self.p_rot_per_axis, 
                                  p_independent_scale_per_axis=self.p_independent_scale_per_axis,
                                  enable_uncertainty=self.enable_uncertainty)
        data_dict[self.data_key] = ret_val[0]
        if seg_list is not None:
            for i in range(len(seg_list)):
                data_dict[self.label_key[i]] = ret_val[1][i]

        return data_dict


def get_training_transforms(
        patch_size,
        rotation_for_DA,
        deep_supervision_scales,
        mirror_axes,
        do_dummy_2d_data_aug,
        order_resampling_data = 3,
        order_resampling_seg = 1,
        border_val_seg = -1,
        use_mask_for_norm = None,
        is_cascaded = False,
        foreground_labels = None,
        regions = None,
        ignore_label = None,
        enable_spatial = True,
        enable_uncertainty = False,
        extra_keys = ['seg', 'seg_sr', 'uncertainty']
    ):
        tr_transforms = []
        ignore_axes = (0,)
        if enable_spatial:
            if do_dummy_2d_data_aug:
                tr_transforms.append(Convert3DTo2DTransform(apply_to_keys=['data',] + extra_keys))
                patch_size_spatial = patch_size[1:]
            else:
                patch_size_spatial = patch_size
                ignore_axes = None

            tr_transforms.append(MySpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=None,
                do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
                do_rotation=True, angle_x=rotation_for_DA['x'], angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
                p_rot_per_axis=1,  # todo experiment with this
                do_scale=True, scale=(0.7, 1.4),
                border_mode_data="constant", border_cval_data=0, order_data=order_resampling_data,
                border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=order_resampling_seg,
                random_crop=False,  # random cropping is part of our dataloaders
                label_key=extra_keys,
                p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
                independent_scale_for_each_axis=False,  # todo experiment with this
                enable_uncertainty=enable_uncertainty
            ))

            if do_dummy_2d_data_aug:
                tr_transforms.append(Convert2DTo3DTransform(apply_to_keys=['data',] + extra_keys,))

        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
        tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                                   p_per_channel=0.5))
        tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
        tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
        tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                            p_per_channel=0.5,
                                                            order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                            ignore_axes=ignore_axes))
        tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
        tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

        if mirror_axes is not None and len(mirror_axes) > 0:
            tr_transforms.append(MirrorTransform(mirror_axes))

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            tr_transforms.append(MaskTransform([i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                                               mask_idx_in_seg=0, set_outside_to=0))


        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            tr_transforms.append(MoveSegAsOneHotToData(1, foreground_labels, 'seg', 'data'))
            tr_transforms.append(ApplyRandomBinaryOperatorTransform(
                channel_idx=list(range(-len(foreground_labels), 0)),
                p_per_sample=0.4,
                key="data",
                strel_size=(1, 8),
                p_per_label=1))
            tr_transforms.append(
                RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                    channel_idx=list(range(-len(foreground_labels), 0)),
                    key="data",
                    p_per_sample=0.2,
                    fill_with_other_class_p=0,
                    dont_do_if_covers_more_than_x_percent=0.15))

        # tr_transforms.append(RenameTransform('seg', 'target', True))

        if regions is not None:
            # the ignore label must also be converted
            tr_transforms.append(ConvertSegmentationToRegionsTransform(list(regions) + [ignore_label]
                                                                       if ignore_label is not None else regions,
                                                                       'seg', 'seg'))

        if deep_supervision_scales is not None:
            tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='seg',
                                                              output_key='seg'))
        tr_transforms.append(NumpyToTensor(['data',] + extra_keys, 'float'))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms

def calculate_dice(prediction, ground_truth, smooth=1e-5):
    prediction = prediction.flatten()
    ground_truth = ground_truth.flatten()
    intersection = np.sum(prediction * ground_truth)
    return (2. * intersection + smooth) / (np.sum(prediction) + np.sum(ground_truth) + smooth)

def evaluate_case(model, case_img, case_label, slice_separation, patch_size, get_HR_results=False):
    model.eval()
    torch.cuda.empty_cache()
    lr_data, properties = preprocess_image(case_img)
    lr_label, properties_label = preprocess_image(case_label, apply_norm=False)
    lr_data, slicer_revert_padding = pad_nd_image(lr_data, patch_size, 'constant', {'value': 0}, True, None)

    with torch.no_grad():
        with torch.autocast('cuda', enabled=True):
            # if input_image is smaller than tile_size we need to pad it to tile_size.
            # lr_data, slicer_revert_padding = pad_nd_image(lr_data, patch_size,
            #                                             'constant', {'value': 0}, True,
            #                                             None)

            slicers = _internal_get_sliding_window_slicers(lr_data.shape[1:], patch_size=patch_size)

            # we need to try except here because we can run OOM in which case we need to fall back to CPU as a results device
            predicted_logits = _internal_predict_sliding_window_return_logits(lr_data, slicers, model, True, 0, 1, patch_size, use_gaussian=True, deep_supervision=False)

            # revert padding
            # predicted_logits = predicted_logits[tuple([slice(None), *slicer_revert_padding[1:]])]
    prediction = predicted_logits.to('cpu') # shape [2, 20, 455, 633]
    prediction = prediction[tuple([slice(None), *slicer_revert_padding[1:]])].squeeze(0)
    with torch.no_grad():
        prediction_prob = torch.softmax(prediction.float(), dim=0).numpy()
    del prediction
    prediction_lr = prediction_prob.argmax(0).astype('uint8') # shape [20, 455, 633]
    dice_lr = calculate_dice(prediction_lr, lr_label.squeeze(0).numpy().astype('uint8'))

    if get_HR_results:
        with torch.no_grad():
            # if input_image is smaller than tile_size we need to pad it to tile_size.
            # lr_data, slicer_revert_padding = pad_nd_image(lr_data, patch_size,
            #                                             'constant', {'value': 0}, True,
            #                                             None)

            slicers = _internal_get_sliding_window_slicers(lr_data.shape[1:], patch_size=patch_size)

            # we need to try except here because we can run OOM in which case we need to fall back to CPU as a results device
            predicted_logits = _internal_predict_sliding_window_return_logits(lr_data, slicers, model, True, 1, slice_separation, [patch_size[0]*slice_separation, patch_size[1], patch_size[2]])

            # revert padding
            # predicted_logits = predicted_logits[tuple([slice(None), *slicer_revert_padding[1:]])]
        prediction = predicted_logits.to('cpu') # shape [2, 20, 455, 633]
        prediction_hr = torch.argmax(prediction, dim=0).squeeze(0).numpy().astype('uint8') # shape [20, 455, 633]
    else:
        prediction_hr = prediction_lr

    return prediction_lr, prediction_hr, lr_label, dice_lr

class _AbstractDiceLoss(torch.nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = torch.nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = torch.nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))

class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    The input to the loss function is assumed to be a logit and will be normalized by the Sigmoid function.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super().__init__(weight, normalization)

    def dice(self, input, target, weight):
        return compute_per_channel_dice(input, target, weight=self.weight)

class BCEDiceLoss(torch.nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, alpha, beta):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.beta = beta
        self.dice = DiceLoss()

    def forward(self, input, target):
        return self.alpha * self.bce(input, target) + self.beta * self.dice(input, target)