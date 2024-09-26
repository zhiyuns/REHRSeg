import numpy as np
import torch
from torch.utils.data import Dataset
from math import ceil
import random
import os
import time
import torch.nn.functional as F
from resize.pytorch import resize

from .augmentations import augment_3d_image
from .patch_ops import get_patch, get_random_centers
from .pad import target_pad
from .parse_image_file import lr_axis_to_z, normalize
from .timer import timer_context

from utils.parse_image_file import parse_image
from utils.blur_kernel_ops import calc_extended_patch_size, parse_kernel

from utils.seg_utils import get_training_transforms, zeroone_normalization, zscore_normalization

class TrainSetMultipleSegSREfficient(Dataset):
    def __init__(
            self,
            image_path,
            split_subjects,
            slice_thickness,
            target_thickness,
            patch_size_ori,
            target_patch_size,
            random_flip=False,
            uncertainty=False,
            preload=True,
            norm=True,
    ):
        self.image_path = image_path
        self.patch_size = patch_size_ori
        self.slice_thickness = slice_thickness
        self.target_thickness = target_thickness
        self.separation = int(slice_thickness / target_thickness)
        self.random_flip = random_flip
        self.uncertainty = uncertainty
        self.norm = norm
        
        self.imgs = []
        self.labels = []
        self.uncertainties = []
        for each_subject in split_subjects:
            if self.uncertainty:
                img_hr, label_hr, uncertainty_hr = self.load_img(os.path.join(image_path, each_subject+'_0000.h5'))
            else:
                img_hr, label_hr, _ = self.load_img(os.path.join(image_path, each_subject+'_0000.h5'))
                uncertainty_hr = [None]
            if preload:
                self.imgs.append(img_hr[:])
                self.labels.append(label_hr[:])
                self.uncertainties.append(uncertainty_hr[:])
            else:
                self.imgs.append(img_hr)
                self.labels.append(label_hr)
                self.uncertainties.append(uncertainty_hr)

        print("Total subjects", len(self.imgs))
        rotation_for_DA = {'x': (-np.pi, np.pi), 'y': (0, 0), 'z': (0, 0)}
        enable_deep_supervision = False
        if enable_deep_supervision:
            deep_supervision_scales = [[1.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.25, 0.25], [0.5, 0.125, 0.125], [0.25, 0.0625, 0.0625], [0.25, 0.03125, 0.03125]]
        else:
            deep_supervision_scales = None
        # mirror_axes = (0, 1, 2)
        mirror_axes = None
        do_dummy_2d_data_aug = True
        use_mask_for_norm = [False]
        is_cascaded = False
        foreground_labels = [1]
        # the input of the transforms should be [bs, 1, 14, 451, 451]??
        self.train_transform = get_training_transforms(target_patch_size[::-1], rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
                order_resampling_data=3, order_resampling_seg=1,
                use_mask_for_norm=use_mask_for_norm,
                is_cascaded=is_cascaded, foreground_labels=foreground_labels,
                regions=None,
                ignore_label=None,
                enable_uncertainty=self.uncertainty,
                extra_keys=['seg', 'seg_sr', 'uncertainty'] if self.uncertainty else ['seg', 'seg_sr'])
    
    def load_img(self, each_subject):
        image, slice_separation, _, blur_fwhm, _, _, _, _ = parse_image(
            each_subject, self.slice_thickness, self.target_thickness
        )
        
        if each_subject.endswith(".h5"):
            img_hr = image['img']
            label_hr = image['seg']
            uncertainty_hr = image['uncertainty']
        else:
            pass # not implemented yet 
        return img_hr, label_hr, uncertainty_hr
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, i):
        img = self.imgs[i][:]
        label = self.labels[i][:]

        if self.norm:
            img = zscore_normalization(img)
        
        x_0, y_0, z_0 = random.randint(0, max(img.shape[0] - self.patch_size[0], 0)), random.randint(0, max(img.shape[1] - self.patch_size[1], 0)), random.randint(0, max(img.shape[2] - self.patch_size[2] * self.separation, 0))

        img = img[x_0:x_0 + self.patch_size[0], y_0:y_0 + self.patch_size[1], z_0:z_0 + self.patch_size[2] * self.separation]
        target_shape = [
            max(s,p) for s, p in zip(img.shape, (self.patch_size[0], self.patch_size[1], self.patch_size[2] * self.separation))
        ]
        img, _ = target_pad(img, target_shape, mode="constant")
        label = label[x_0:x_0 + self.patch_size[0], y_0:y_0 + self.patch_size[1], z_0:z_0 + self.patch_size[2] * self.separation]
        label, _ = target_pad(label, target_shape, mode="constant")
        if self.uncertainty:
            uncertainty = self.uncertainties[i][:]
            uncertainty = uncertainty[x_0:x_0 + self.patch_size[0], y_0:y_0 + self.patch_size[1], z_0:z_0 + self.patch_size[2] * self.separation]
            uncertainty, _ = target_pad(uncertainty, target_shape, mode="constant")

        if self.random_flip:
            if random.random() < 0.5:
                img = np.flip(img, axis=0)
                label = np.flip(label, axis=0)
                uncertainty = np.flip(uncertainty, axis=0) if self.uncertainty else None
            if random.random() < 0.5:
                img = np.flip(img, axis=1)
                label = np.flip(label, axis=1)
                uncertainty = np.flip(uncertainty, axis=1) if self.uncertainty else None
            if random.random() < 0.5:
                img = np.flip(img, axis=2)
                label = np.flip(label, axis=2)
                uncertainty = np.flip(uncertainty, axis=2) if self.uncertainty else None
        
        img = img[:,:,::self.separation]
        label_lr = label[:,:,::self.separation]

        img = img.copy().transpose(2,1,0)[None, None, ...] # _, channel, z, x, y
        label = label.copy().transpose(2,1,0)[None, None, ...] # _, channel, z, x, y
        # uncertainty = uncertainty.copy().transpose(2,1,0)[None, None, ...] # _, channel, z, x, y
        label_lr = label_lr.copy().transpose(2,1,0)[None, None, ...] # _, channel, z, x, y
        
        if self.uncertainty:
            uncertainty_lr = uncertainty[:,:,::self.separation]
            uncertainty_lr = uncertainty_lr.copy().transpose(2,1,0)[None, None, ...] # _, channel, z, x, y
            uncertainty_lr = 1 - uncertainty_lr / 255. * 0.99
            out_data = self.train_transform(**{'data': img.astype('float32'), 'seg': label_lr, 'seg_sr': label, 'uncertainty': uncertainty_lr})
            uncertainty_lr = out_data['uncertainty'].squeeze(0)
        else:
            out_data = self.train_transform(**{'data': img.astype('float32'), 'seg': label_lr, 'seg_sr': label})
            uncertainty_lr = 0

        img = out_data['data'].squeeze(0)
        label_lr = out_data['seg'].squeeze(0)
        label = out_data['seg_sr'].squeeze(0)
        
        return img, label_lr, label, uncertainty_lr

class TrainSetMultipleSegSR(Dataset):
    def __init__(
            self,
            image_path,
            split_subjects,
            slice_thickness,
            target_thickness,
            patch_size,
            random_flip=False,
    ):
        if len(patch_size) == 2:
            patch_size = (*patch_size, 1)
        self.patch_size = patch_size
        self.random_flip = random_flip

        self.split_subjects = split_subjects
        self.imgs = []
        self.labels = []
        for each_subject in split_subjects:
            image, slice_separation, _,  blur_fwhm, _, _, _, _ = parse_image(
                os.path.join(image_path, each_subject+'_0000.nii.gz'), slice_thickness, target_thickness
            )
            image = image.squeeze() # shape (x, y, z, 2)
            if len(image.shape) == 3:
                image = image[..., np.newaxis]
            target_shape = [
                max(s,p) for s, p in zip(image.shape[:3], (self.patch_size[0], self.patch_size[1], self.patch_size[2]))
            ] + [image.shape[3], 2]
            # image, _ = target_pad(image, target_shape, mode="reflect")
            image, _ = target_pad(image, target_shape, mode="constant")
            print(each_subject, "image shape", image.shape)
            self.imgs.append(image[...,:1])
            self.labels.append(image[...,1:].astype('uint8'))
        print("Total subjects", len(self.imgs))
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, i):
        img = self.imgs[i]
        label = self.labels[i]

        x_0, y_0, z_0 = random.randint(0, img.shape[0] - self.patch_size[0]), random.randint(0, img.shape[1] - self.patch_size[1]), random.randint(0, img.shape[2] - self.patch_size[2])
        img = img[x_0:x_0 + self.patch_size[0], y_0:y_0 + self.patch_size[1], z_0:z_0 + self.patch_size[2], :]
        label = label[x_0:x_0 + self.patch_size[0], y_0:y_0 + self.patch_size[1], z_0:z_0 + self.patch_size[2], :].astype('float32')
        if label.max() > 1:
            print(self.split_subjects[i])
        if self.random_flip:
            if random.random() < 0.5:
                img = np.flip(img, axis=0)
                label = np.flip(label, axis=0)
            if random.random() < 0.5:
                img = np.flip(img, axis=1)
                label = np.flip(label, axis=1)
            if random.random() < 0.5:
                img = np.flip(img, axis=2)
                label = np.flip(label, axis=2)

        img = torch.from_numpy(img.copy().transpose(3,2,1,0)) # channel, z, x, y
        label = torch.from_numpy(label.copy().transpose(3,2,1,0)) # channel, z, x, y

        return img, label


class TrainSetMultiple(Dataset):
    def __init__(
            self,
            image_path,
            split_subjects,
            slice_thickness,
            target_thickness,
            blur_kernel_fpath,
            blur_kernel_name,
            patch_size,
            random_flip,
            device,
            preload=True,
            blur=True,
            nnunet_transform=False,
            norm=True,
    ):
        if len(patch_size) == 2:
            patch_size = (*patch_size, 1)
        self.patch_size = patch_size
        self.random_flip = random_flip
        self.device = device
        self.preload = preload
        self.blur = blur

        all_subject_names = os.listdir(image_path)
        self.image_path = image_path
        self.all_subjects = split_subjects
        self.slice_thickness = slice_thickness
        self.target_thickness = target_thickness
        self.blur_kernel_fpath = blur_kernel_fpath
        self.blur_kernel_name = blur_kernel_name
        self.slice_separation = float(slice_thickness / target_thickness)

        if nnunet_transform:
            rotation_for_DA = {'x': (-np.pi, np.pi), 'y': (0, 0), 'z': (0, 0)}
            deep_supervision_scales = None
            # mirror_axes = (0, 1, 2)
            mirror_axes = None
            do_dummy_2d_data_aug = True
            use_mask_for_norm = [False]
            is_cascaded = False
            foreground_labels = [1]
            # the input of the transforms should be [bs, 1, 14, 451, 451]??
            self.train_transform = get_training_transforms(patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
                    order_resampling_data=3, order_resampling_seg=1,
                    use_mask_for_norm=use_mask_for_norm,
                    is_cascaded=is_cascaded, foreground_labels=foreground_labels,
                    regions=None,
                    ignore_label=None,
                    enable_spatial=False,
                    enable_uncertainty=self.blur,
                    extra_keys=['seg', 'img_lr'] if self.blur else ['seg',])
        else:
            self.train_transform = None
        
        self.imgs_hr = []
        self.labels_hr = []
        self.imgs_filtered_x = []
        self.imgs_filtered_y = []
        for each_subject in self.all_subjects:
            each_subject = [x for x in all_subject_names if each_subject in x][0]
            img_hr, label_hr, image_x_rgb, image_y_rgb = self.load_img(each_subject, filter_x=self.blur, filter_y=self.blur)
            print(each_subject, "image shape", img_hr.shape)
            if self.preload:
                self.imgs_hr.append(img_hr[:])
                self.labels_hr.append(label_hr[:])
                self.imgs_filtered_x.append(image_x_rgb[:])
                self.imgs_filtered_y.append(image_y_rgb[:])
            else:
                self.imgs_hr.append(img_hr)
                self.labels_hr.append(label_hr)
                self.imgs_filtered_x.append(image_x_rgb)
                self.imgs_filtered_y.append(image_y_rgb)

    def __len__(self):
        return len(self.all_subjects)
    
    def load_img(self, each_subject, filter_x=True, filter_y=True):
        image, slice_separation, lr_axis,  blur_fwhm, _, _, _, _ = parse_image(
            os.path.join(self.image_path, each_subject), self.slice_thickness, self.target_thickness
        )
        
        if each_subject.endswith(".h5"):
            img_hr = image['img_hr']
            label_hr = image['label_hr']
            image_x_rgb = image['image_x_rgb'] if filter_x else [None]
            image_y_rgb = image['image_y_rgb'] if filter_y else [None]
        else:
            image = image.squeeze() # shape (x, y, z, 2)
            if len(image.shape) == 3:
                image = image[..., np.newaxis]
                # image = lr_axis_to_z(image, lr_axis)
            blur_kernel = parse_kernel(self.blur_kernel_fpath, self.blur_kernel_name, blur_fwhm)
            img_hr = image[...,:1]
            label_hr = image[...,1:].astype('uint8')
            
            if filter_x:
                image_x = torch.from_numpy(image.transpose(2, 3, 0, 1)) # z, channel, x, y
                image_x_rgb = image_x[:, 0:1, ...]
                image_x_rgb = F.conv2d(image_x_rgb, blur_kernel, padding="same").numpy()
            else:
                image_x_rgb = [None]

            if filter_y:
                image_y = torch.from_numpy(image.transpose(2, 3, 1, 0)) # z, channel, y, x
                image_y_rgb = image_y[:, 0:1, ...]
                image_y_rgb = F.conv2d(image_y_rgb, blur_kernel, padding="same").numpy()
            else:
                image_y_rgb = [None]
        return img_hr, label_hr, image_x_rgb, image_y_rgb

    def __getitem__(self, i):
        img_hr = self.imgs_hr[i]
        label_hr = self.labels_hr[i]
        if self.blur:
            if random.random() < 0.5:
                img_hr = np.transpose(img_hr[:], (1, 0, 2, 3))
                label_hr = np.transpose(label_hr[:], (1, 0, 2, 3))
                img_lr = self.imgs_filtered_y[i]
            else:
                img_lr = self.imgs_filtered_x[i]
        else:
            if random.random() < 0.5:
                img_hr = np.transpose(img_hr[:], (1, 0, 2, 3))
                label_hr = np.transpose(label_hr[:], (1, 0, 2, 3))
            
        slice_separation = self.slice_separation
        x_0, y_0, z_0 = random.randint(0, max(img_hr.shape[0] - self.patch_size[0], 0)), random.randint(0, max(img_hr.shape[1] - self.patch_size[1], 0)), random.randint(0, max(img_hr.shape[2] - self.patch_size[2], 0))
        img_hr = img_hr[x_0:x_0 + self.patch_size[0], y_0:y_0 + self.patch_size[1], z_0:z_0 + self.patch_size[2], :]
        patch_label_hr = label_hr[x_0:x_0 + self.patch_size[0], y_0:y_0 + self.patch_size[1], z_0:z_0 + self.patch_size[2], :].astype('float32')
        img_hr = img_hr.transpose(2, 3, 0, 1) # z, channel, x, y
        patch_label_hr = patch_label_hr.transpose(2, 3, 0, 1) # z, channel, x, y

        # pad to target
        target_shape = [
            max(s,p) for s, p in zip(img_hr.shape, (self.patch_size[2], 1, self.patch_size[0], self.patch_size[0]))
        ]
        img_hr, _ = target_pad(img_hr, target_shape, mode="constant")
        patch_label_hr, _ = target_pad(patch_label_hr, target_shape, mode="constant")

        if self.train_transform is not None:
            if self.blur:
                img_lr = img_lr[z_0:z_0 + self.patch_size[2], :, x_0:x_0 + self.patch_size[0], y_0:y_0 + self.patch_size[1]]
                img_lr, _ = target_pad(img_lr, target_shape, mode="constant")
                out_data = self.train_transform(**{'data': img_hr.transpose(1,0,2,3)[None, ...], 
                                                'seg': patch_label_hr.transpose(1,0,2,3)[None, ...], 
                                                'img_lr': img_lr.transpose(1,0,2,3)[None, ...]})
                img_hr = out_data['data'].squeeze(0).permute(1,0,2,3)
                patch_label_hr = out_data['seg'].squeeze(0).permute(1,0,2,3)
                img_lr = out_data['img_lr'].squeeze(0).permute(1,0,2,3)
            else:
                out_data = self.train_transform(**{'data': img_hr.transpose(1,0,2,3)[None, ...], 
                                                'seg': patch_label_hr.transpose(1,0,2,3)[None, ...]})
                img_hr = out_data['data'].squeeze(0).permute(1,0,2,3)
                patch_label_hr = out_data['seg'].squeeze(0).permute(1,0,2,3)
                img_lr = img_hr.detach()
        else:
            img_hr = torch.from_numpy(img_hr)
            patch_label_hr = torch.from_numpy(patch_label_hr)
            if self.blur:
                img_lr = img_lr[z_0:z_0 + self.patch_size[2], :, x_0:x_0 + self.patch_size[0], y_0:y_0 + self.patch_size[1]]
                img_lr, _ = target_pad(img_lr, target_shape, mode="constant")
                img_lr = torch.from_numpy(img_lr)
            else:
                img_lr = img_hr.detach()

        img_hr = torch.cat((img_hr, patch_label_hr), dim=1)
        
        # simulate lr image
        img_lr = resize(img_lr, (slice_separation, 1), order=3)
        label_lr = resize(patch_label_hr, (slice_separation, 1), order=0)
        img_lr = torch.cat((img_lr, label_lr), dim=1)
        
        img_hr = img_hr.permute(1, 2, 0, 3) # channel, x, z, y
        img_lr = img_lr.permute(1, 2, 0, 3)

        if img_hr.shape[2] > 1 and random.random() < 0.1:
            zero_slice = torch.zeros_like(img_lr[:,0:1])
            img_lr[:,0:1] = zero_slice

        if img_hr.shape[2] > 1 and random.random() < 0.1:
            zero_slice = torch.zeros_like(img_lr[:,-1:])
            img_lr[:,-1:] = zero_slice
        
        if self.random_flip:
            if random.random() < 0.5:
                # flip along the x-axis
                img_hr = img_hr.flip(1)
                img_lr = img_lr.flip(1)
            if random.random() < 0.5:
                # flip along the z-axis
                img_hr = img_hr.flip(2)
                img_lr = img_lr.flip(2)
            if random.random() < 0.5:
                # flip along the y-axis
                img_hr = img_hr.flip(3)
                img_lr = img_lr.flip(3)

        if random.random() < 0.5:
            img_hr = img_hr.permute(0, 1, 3, 2)
            img_lr = img_lr.permute(0, 1, 3, 2)
            img_hr = img_hr.squeeze(3) # squeeze for 2D image
            img_lr = img_lr.squeeze(3)
        else:
            img_hr = img_hr.squeeze(2) # squeeze for 2D image
            img_lr = img_lr.squeeze(2)
        # patch_hr = patch_hr.squeeze(0)
        # patch_lr = patch_lr.squeeze(0)
        return img_lr, img_hr


class TrainSet(Dataset):
    def __init__(
            self,
            image,
            slice_separation,
            patch_size,
            ext_patch_crop,
            device,
            blur_kernel,
            n_patches,
            patch_sampling,
            verbose=True,
    ):
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.slice_separation = slice_separation

        self.ext_patch_crop = ext_patch_crop
        self.device = device
        self.blur_kernel = blur_kernel

        with timer_context(
                "Gathering data augmentations (flips and transposition in-plane)...",
                verbose=verbose,
        ):
            imgs_hr = [image, np.transpose(image, (1, 0, 2, 3))]

        with timer_context(
                "Padding image out to extract patches correctly...", verbose=verbose
        ):
            self.imgs_hr = []
            self.pads = []

            for image in imgs_hr:
                # Pad out s.t. in-planes are at least the patch size in each direction
                target_shape = [
                                   s + p for s, p in zip(image.shape[:-1], self.patch_size[:-1])
                               ] + [image.shape[2], 2]

                # apply the pad
                image, pads = target_pad(image, target_shape, mode="reflect")
                self.imgs_hr.append(image)
                self.pads.append(pads)

        with timer_context(
                "Generating (weighted) random patch centers..", verbose=verbose
        ):
            if patch_sampling == "uniform":
                weighted = False
            elif patch_sampling == "gradient":
                weighted = True
            self.centers = get_random_centers(
                self.imgs_hr,
                self.patch_size,
                self.n_patches,
                weighted=weighted,
            )

    def __len__(self):
        return self.n_patches

    def __getitem__(self, i):
        # Pull the HR patch
        aug_idx, center_idx = self.centers[i]
        img_hr = self.imgs_hr[aug_idx]
        patch_hr = get_patch(img_hr, center_idx, self.patch_size)

        patch_hr = torch.from_numpy(patch_hr.transpose(2, 0, 1))
        patch_hr = augment_3d_image(patch_hr)

        patch_hr = patch_hr.unsqueeze(0)
        patch_lr_rgb = patch_hr[:, 0:1, ...]
        patch_lr_rgb = F.conv2d(patch_lr_rgb, self.blur_kernel, padding="same")
        patch_lr = torch.cat((patch_lr_rgb, patch_hr[:, 1:2, ...]), dim=1)

        patch_hr = patch_hr[self.ext_patch_crop]
        patch_lr = patch_lr[self.ext_patch_crop]

        # Downsample the LR patches
        patch_lr = resize(patch_lr, (self.slice_separation, 1), order=3)

        patch_hr = patch_hr.squeeze(0)
        patch_lr = patch_lr.squeeze(0)
        return patch_lr, patch_hr
