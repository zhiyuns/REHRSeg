import argparse
import sys
import os
from pathlib import Path
import time
import json
import torch
from torch.utils.data import DataLoader
from math import ceil

from models.wdsr import WDSR
from models.FLAVR.FLAVR_arch import UNet_3D_3D

from utils.train_set import TrainSetMultiple
from utils.timer import timer_context
from utils.parse_image_file import parse_image
from utils.misc_utils import parse_device, LossProgBar
from utils.blur_kernel_ops import calc_extended_patch_size, parse_kernel
from utils.seg_utils import BCEDiceLoss

# Optimize torch
# noinspection PyUnresolvedReferences
torch.backends.cudnn.benchmark = True

import torch

import SimpleITK as sitk
from resize.pytorch import resize

def nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None): # pylint: disable=redefined-builtin
    assert isinstance(input, torch.Tensor)
    if posinf is None:
        posinf = torch.finfo(input.dtype).max
    if neginf is None:
        neginf = torch.finfo(input.dtype).min
    assert nan == 0
    return torch.clamp(input.unsqueeze(0).nansum(0), min=neginf, max=posinf, out=out)

def set_random_seed(seed):
    # Set the random seed for CPU
    torch.manual_seed(seed)

    # Set the random seed for all GPUs (if using CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        # Enable deterministic algorithms for cuDNN operations
        torch.backends.cudnn.deterministic = True

        # Disable cuDNN benchmark mode
        torch.backends.cudnn.benchmark = False

def separate_weight_decayable_params(model, freeze_others=False):
    decay_params = []
    other_params = []
    for pname, p in model.named_parameters():
        if 'encoder.stem.0' in pname or 'outconv.1' in pname or 'feature_fuse' in pname or 'uncertainty' in pname:
            print(pname)
            decay_params += [p]
        else:
            other_params += [p]
            if freeze_others:
                p.requires_grad = False
    
    if freeze_others:
        return decay_params
    else:
        params = [
            {'params': decay_params, 'weight_decay': 0.001},
            {'params': other_params, 'lr': 0.00001, 'weight_decay': 0.0},
        ]
        return params

# Example usage
set_random_seed(0)


def main(args=None):
    #################### ARGUMENTS ####################
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-fpath", type=str, required=True)
    parser.add_argument("--weight-dir", type=str, required=True)
    parser.add_argument("--gpu-id", type=int, default=-1)
    parser.add_argument("--interp-order", type=int, default=3)
    parser.add_argument("--fold", type=str, default='all')
    parser.add_argument(
        "--n-patches", type=int, default=832000
    )  # The sum of 4 phases from Shuo's thesis
    parser.add_argument(
        "--save-iters", type=int, default=10000
    )  # The sum of 4 phases from Shuo's thesis
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sr_mode", type=str, default='img+seg', choices=['img', 'seg', 'img+seg'])
    parser.add_argument("--weight-seg", type=float, default=1.0)
    parser.add_argument("--num-slices", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-blocks", type=int, default=16)
    parser.add_argument("--num-channels", type=int, default=32)
    parser.add_argument("--slice-thickness", type=float)
    parser.add_argument("--target-thickness", type=float)
    parser.add_argument("--blur-kernel", type=str, default="rf-pulse-slr")
    parser.add_argument("--blur-kernel-fpath", type=str)
    parser.add_argument("--patch-sampling", type=str, default="gradient")
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--pretrain-path", type=str, default=None)
    parser.add_argument("--random-flip", action="store_true", default=False)
    parser.add_argument("--preload", action="store_true", default=False)
    parser.add_argument("--enable-uncertainty", action="store_true", default=False)
    parser.add_argument("--no-blur", action="store_true", default=False)
    parser.add_argument("--nnunet-transform", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)

    args = parser.parse_args(args if args is not None else sys.argv[1:])
    split_path = r'../../nnUNet/DATASET/nnUNet_preprocessed/Dataset506_BoneTumor/splits_final.json'

    if not Path(args.in_fpath).exists():
        raise ValueError("Input image path does not exist.")
    if args.blur_kernel_fpath is not None and not Path(args.blur_kernel_fpath).exists():
        raise ValueError("Blur kernel fpath is specified but does not exist.")

    # A nice print statement divider for the command line
    text_div = "=" * 10

    print(f"{text_div} BEGIN TRAINING {text_div}")

    
    weight_dir = Path(args.weight_dir)
    n_steps = int(ceil(args.n_patches / args.batch_size))
    learning_rate = args.lr
    device = parse_device(args.gpu_id)

    if not weight_dir.exists():
        weight_dir.mkdir(parents=True)

    slice_separation = float(args.slice_thickness / args.target_thickness)
    if args.num_slices > 1:
        lr_patch_size = [args.num_slices, args.patch_size, args.patch_size]
        model = UNet_3D_3D(
            img_channels=len(args.sr_mode.split('+')),
            block="unet_18",
            n_inputs=args.num_slices,
            n_outputs=int(slice_separation),
            batchnorm=False,
            joinType="concat",
            upmode="transpose",
            use_uncertainty=args.enable_uncertainty
        ).to(device)
        
    else:
        lr_patch_size = [args.patch_size, args.patch_size]
        model = WDSR(
            out_channel=len(args.sr_mode.split('+')),
            n_resblocks=args.num_blocks,
            num_channels=args.num_channels,
            scale=slice_separation,
        ).to(device)
    
    if args.pretrain_path is not None:
        pretrained_dict = torch.load(args.pretrain_path)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
            for key in list(pretrained_dict.keys()):
                if "module" in key:
                    pretrained_dict[key.replace("module.", "")] = pretrained_dict.pop(key)
                if 'encoder.stem.0' in key or 'outconv.1' in key or 'feature_fuse' in key:
                    pretrained_dict.pop(key.replace("module.", ""))
        if 'model' in pretrained_dict:
            pretrained_dict = pretrained_dict['model']
            for key in list(pretrained_dict.keys()):
                if "module" in key:
                    pretrained_dict[key.replace("module.", "")] = pretrained_dict.pop(key)
                if 'outconv.1' in key or 'feature_fuse' in key:
                    pretrained_dict.pop(key.replace("module.", ""))
        
        model.load_state_dict(pretrained_dict, strict=False)
    if args.resume_path is not None:
        model.load_state_dict(torch.load(args.resume_path)["model"])

    patch_size = model.calc_out_patch_size(lr_patch_size)
    
    if args.fold == 'all':
        split_data = os.listdir(args.in_fpath)
    else:
        with open(split_path, 'r') as f:
            split_data = json.load(f)[args.fold]['train']
    
    with timer_context("Parsing image file...", verbose=args.verbose):
        train_dataset = TrainSetMultiple(args.in_fpath, split_data, args.slice_thickness, args.target_thickness, args.blur_kernel_fpath, args.blur_kernel, patch_size, args.random_flip, device, preload=args.preload, blur=not args.no_blur, nnunet_transform=args.nnunet_transform)

    # ===== MODEL SETUP =====

    # opt = torch.optim.Adam(separate_weight_decayable_params(model), betas=(0.9, 0.99), lr=learning_rate)

    opt = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=learning_rate)
    start_iter = int(args.resume_path.split('_')[-1].split('.')[0]) if args.resume_path is not None else 0
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=opt,
        max_lr=learning_rate,
        total_steps=n_steps + start_iter,
        cycle_momentum=True,
    )
    
    
    opt.step()  # necessary for the LR scheduler
    
    for _ in range(start_iter):
        scheduler.step()

    loss_obj = torch.nn.L1Loss().to(device)
    # loss_seg = torch.nn.L1Loss().to(device)
    loss_seg = BCEDiceLoss(alpha=1, beta=1).to(device)


    # ===== LOAD AND PROCESS DATA =====

    data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=0,
    )

    # ===== TRAIN =====
    print(f"\n{text_div} TRAINING NETWORK {text_div}\n")
    train_st = time.time()

    loss_names = ["loss", "lr"]
    total_iters = 0

    with LossProgBar(args.n_patches, args.batch_size, loss_names) as pbar:
        while True:
            for i, (patches_lr, patches_hr) in enumerate(data_loader):
                patches_hr = patches_hr.to(device)
                patches_lr = patches_lr.to(device)

                '''
                # save for visualize
                for i in range(patches_hr.shape[0]):
                    sitk.WriteImage(sitk.GetImageFromArray(patches_hr[i][0].squeeze().cpu().numpy()), f"patches_hr_img_{i}.nii.gz")
                    sitk.WriteImage(sitk.GetImageFromArray(patches_lr[i][0].squeeze().cpu().numpy()), f"patches_lr__img{i}.nii.gz")
                    sitk.WriteImage(sitk.GetImageFromArray(patches_hr[i][1].squeeze().cpu().numpy()), f"patches_hr_seg_{i}.nii.gz")
                    sitk.WriteImage(sitk.GetImageFromArray(patches_lr[i][1].squeeze().cpu().numpy()), f"patches_lr_seg_{i}.nii.gz")
                '''
                
                if args.num_slices > 1:
                    patches_hr = patches_hr[:,:, int(slice_separation)*(args.num_slices//2-1):int(slice_separation)*(args.num_slices//2), ...]
                        
                if args.sr_mode == 'img':
                    patches_hr = patches_hr[:,0:1,...]
                    patches_lr = patches_lr[:,0:1,...]
                    if args.enable_uncertainty:
                        patches_hr_hat, uncertainty = model(patches_lr)
                        loss = loss_obj(patches_hr_hat, patches_hr)
                        loss += torch.mean(torch.div(torch.abs(patches_hr_hat-patches_hr), uncertainty) + torch.log(uncertainty))
                    else:
                        patches_hr_hat = model(patches_lr)
                        loss = loss_obj(patches_hr_hat, patches_hr)
                elif args.sr_mode == 'seg':
                    patches_hr = patches_hr[:,1:,...]
                    patches_lr = patches_lr[:,1:,...]
                    if args.enable_uncertainty:
                        patches_hr_hat, _ = model(patches_lr)
                    else:
                        patches_hr_hat = model(patches_lr)
                    loss = loss_seg(patches_hr_hat, patches_hr)
                elif args.sr_mode == 'img+seg':
                    if args.enable_uncertainty:
                        patches_hr_hat, uncertainty = model(patches_lr)
                        loss = loss_obj(patches_hr_hat[:,0:1,...], patches_hr[:,0:1,...])
                        loss += torch.mean(torch.div(torch.abs(patches_hr_hat[:,0:1,...]-patches_hr[:,0:1,...]), uncertainty) + torch.log(uncertainty))
                        error_map = torch.abs(patches_hr_hat[:,0:1,...].detach()-patches_hr[:,0:1,...])
                        loss += loss_obj(uncertainty, error_map)
                    else:
                        patches_hr_hat = model(patches_lr)
                        loss = loss_obj(patches_hr_hat[:,0:1,...], patches_hr[:,0:1,...])
                    loss += loss_seg(patches_hr_hat[:,1:,...], patches_hr[:,1:,...]) * args.weight_seg

                
                opt.zero_grad()
                loss.backward()
                # for param in model.parameters():
                #     if param.grad is not None:
                #         nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                opt.step()
                scheduler.step()

                # Progress bar update
                pbar.update({"loss": loss, "lr": torch.Tensor([opt.param_groups[0]["lr"]])})
                if total_iters>0 and total_iters % args.save_iters == 0:
                    weight_path = weight_dir / f"weights_{total_iters}.pt"
                    torch.save({"model": model.state_dict()}, str(weight_path))

                total_iters += 1
            if total_iters >= n_steps:
                break

    # ===== SAVE MODEL CONDITIONS =====
    weight_path = weight_dir / "last_weights.pt"

    torch.save({"model": model.state_dict()}, str(weight_path))

    train_en = time.time()
    print(f"\n\tElapsed time to finish training: {train_en - train_st:.4f}s")


if __name__ == "__main__":
    main()