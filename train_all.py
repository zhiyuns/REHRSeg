import argparse
import sys
import os
from pathlib import Path
import time
import json
import torch
from tqdm import tqdm
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from omegaconf import OmegaConf
from math import ceil
from torch.utils.data import DataLoader
import h5py
import itertools

from torch.optim.lr_scheduler import PolynomialLR

from models.wdsr import WDSR
from models.FLAVR.FLAVR_arch import UNet_3D_3D
from models.seg_model import SegModel, Distiller

from utils.train_set import TrainSetMultiple, TrainSetMultipleSegSREfficient
from utils.timer import timer_context
from utils.parse_image_file import parse_image
from utils.misc_utils import parse_device, LossProgBar
from utils.blur_kernel_ops import calc_extended_patch_size, parse_kernel

from utils.seg_utils import zscore_normalization, BCEDiceLoss, _build_loss, evaluate_case, calculate_dice
from utils.sr_utils import inference_smore, inference_flavr, postprocess_smore


def merge_images_and_labels(main_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_subjects = os.listdir(main_dir)
    print(f'Merging images and labels for a total of {len(all_subjects)} subjects')
    for main_image_name in all_subjects:
        output_path = os.path.join(output_dir, main_image_name)
        if os.path.exists(output_path): continue
        main_image_path = os.path.join(main_dir, main_image_name)
        img_path = main_image_path
        label_path = main_image_path.replace('imagesTr', 'labelsTr').replace('_0000.nii.gz', '.nii.gz')

        if os.path.exists(label_path):
            main_image_data = nib.load(img_path).get_fdata()
            label_data = nib.load(label_path).get_fdata()

            if main_image_data.shape != label_data.shape:
                print(f"Shape mismatch between main image {main_image_name} and label {label_path}")
                continue

            merged_data = np.zeros(main_image_data.shape + (2,), dtype=np.float32)
            merged_data[..., 0] = main_image_data
            merged_data[..., 1] = label_data

            merged_image = nib.Nifti1Image(merged_data, affine=None)
            merged_image.header['pixdim'][1:5] = (1.0, 1.0, 4.0, 1.0)
            
            nib.save(merged_image, output_path)
        else:
            print(f"Segmentation label file not found for {main_image_name}")

def separate_weight_extensive_params(model, freeze_others=False, ori_lr=0.01):
    extensive_params = []
    other_params = []
    for pname, p in model.named_parameters():
        if 'sr_head' in pname:
            print(pname)
            extensive_params += [p]
        else:
            other_params += [p]
            if freeze_others:
                p.requires_grad = False
    
    if freeze_others:
        return extensive_params
    else:
        params = [
            {'params': extensive_params},
            {'params': other_params, 'lr': ori_lr / 10, 'weight_decay': 0.0},
        ]
        return params

def get_intermediate_features(model_sr, img_lr, label_lr, device):
    img_lr = zscore_normalization(img_lr)
    input_data = torch.cat((img_lr, label_lr), dim=1)
    features_sr = {}
    for st in range(0, input_data.shape[2]-1):
        if st == 0:
            batch = input_data[:,:,0:3]
            batch = torch.cat([torch.zeros(batch.shape[0], batch.shape[1], 4-batch.shape[2], *batch.shape[3:]).to(device), batch], dim=2)
        elif st == input_data.shape[2]-2:
            batch = input_data[:,:,st-1:]
            batch = torch.cat([batch, torch.zeros(batch.shape[0], batch.shape[1], 4-batch.shape[2], *batch.shape[3:]).to(device)], dim=2)
        else:
            batch = input_data[:,:,st-1:st+3]
        batch_input = batch.clone()
        features_each = model_sr(batch_input, return_inetermediate_feature=True)
        for i, f in enumerate(features_each):
            if i not in features_sr:
                features_sr[i] = []
            features_sr[i].append(f[:,:,1:2,...])
    # append the last slice feature
    for i, f in enumerate(features_each):
        if i not in features_sr:
            features_sr[i] = []
        features_sr[i].append(f[:,:,2:3,...])
    for i in features_sr:
        features_sr[i] = torch.cat(features_sr[i], dim=2)

    return features_sr

def train_sr(n_patches, batch_size, model, opt, scheduler, data_loader, device, loss_obj, loss_seg, n_steps, slice_separation, num_slices, enable_uncertainty, weight_dir, save_iters):
    total_iters = 0
    with LossProgBar(n_patches, batch_size, ["loss", "lr"]) as pbar:
        while True:
            for i, (patches_lr, patches_hr) in enumerate(data_loader):
                patches_hr = patches_hr.to(device)
                patches_lr = patches_lr.to(device)

                if num_slices > 1:
                    patches_hr = patches_hr[:,:, int(slice_separation)*(num_slices//2-1):int(slice_separation)*(num_slices//2), ...]

                if enable_uncertainty:
                    patches_hr_hat, uncertainty = model(patches_lr)
                    loss = loss_obj(patches_hr_hat[:,0:1,...], patches_hr[:,0:1,...])
                    loss += torch.mean(torch.div(torch.abs(patches_hr_hat[:,0:1,...]-patches_hr[:,0:1,...]), uncertainty) + torch.log(uncertainty))
                    error_map = torch.abs(patches_hr_hat[:,0:1,...].detach()-patches_hr[:,0:1,...])
                    loss += loss_obj(uncertainty, error_map)
                else:
                    patches_hr_hat = model(patches_lr)
                    loss = loss_obj(patches_hr_hat[:,0:1,...], patches_hr[:,0:1,...])
                loss += loss_seg(patches_hr_hat[:,1:,...], patches_hr[:,1:,...]) * 1.0

                opt.zero_grad()
                loss.backward()
                opt.step()
                scheduler.step()

                # Progress bar update
                pbar.update({"loss": loss, "lr": torch.Tensor([opt.param_groups[0]["lr"]])})
                if total_iters>0 and total_iters % save_iters == 0:
                    weight_path = weight_dir / f"weights_{total_iters}.pt"
                    torch.save({"model": model.state_dict()}, str(weight_path))

                total_iters += 1
            if total_iters >= n_steps:
                break
    return 

def evaluate(model_seg, patch_size_ori, val_img_path, val_label_path, split_path, fold, save_path=None, eval_HR=False, seperation=1):
    bad_cases = []
    with open(split_path, 'r') as f:
        split_data = json.load(f)[fold]['val']
    all_dice = []
    all_pred = []
    all_label = []
    for subject in split_data:
        if subject in bad_cases: continue
        test_img = os.path.join(val_img_path, subject+'_0000.nii.gz')
        test_label = os.path.join(val_label_path, subject+'.nii.gz')
        pred_lr, pred_hr, label_lr, dice_lr = evaluate_case(model_seg, test_img, test_label, slice_separation=seperation, patch_size=patch_size_ori[::-1], get_HR_results=eval_HR)
        if save_path is not None:
            os.makedirs(os.path.join(save_path, 'val'), exist_ok=True)
            ori_info = sitk.ReadImage(test_img)
            #copy information
            pred_lr_img = sitk.GetImageFromArray(pred_lr)
            pred_lr_img.SetSpacing(ori_info.GetSpacing())
            pred_lr_img.SetOrigin(ori_info.GetOrigin())
            pred_lr_img.SetDirection(ori_info.GetDirection())
            sitk.WriteImage(pred_lr_img, os.path.join(save_path, 'val', f"{subject}_pred_lr.nii.gz"))
            if eval_HR:
                pred_hr_img = sitk.GetImageFromArray(pred_hr)
                ori_spacing = ori_info.GetSpacing()
                pred_hr_img.SetSpacing([ori_spacing[0], ori_spacing[1], ori_spacing[2]/seperation])
                pred_hr_img.SetOrigin(ori_info.GetOrigin())
                pred_hr_img.SetDirection(ori_info.GetDirection())
                sitk.WriteImage(pred_hr_img, os.path.join(save_path, 'val', f"{subject}_pred_hr.nii.gz"))
        all_pred.append(pred_lr.flatten())
        all_label.append(label_lr.flatten())
        print(f"Subject {subject}: {dice_lr}")
        all_dice.append(dice_lr)
    print(f'Global dice: {calculate_dice(np.concatenate(all_pred), np.concatenate(all_label))}')
    print(f"Average dice: {sum(all_dice)/len(all_dice)}")
    print(f"Std dice: {np.std(all_dice)}")
    print(f"Max dice: {max(all_dice)}")
    print(f"Min dice: {min(all_dice)}")
    val_dice = sum(all_dice)/len(all_dice)
        
    return val_dice

def main(
    data_path: str,
    tmp_path: str,
    checkpoint_path: str,
    seg_path: str,
    smore_initialization: bool,
    pretrain_path: str,
    batch_size_sr: int,
    lr_sr: float,
    n_patches: int,
    save_iters_sr: int,
    save_iters_segsr: int,
    num_slices: int,
    patch_size: int,
    slice_thickness: int,
    target_thickness: int,
    blur_kernel: str,
    random_flip: bool,
    nnunet_transform: bool,
    enable_uncertainty: bool,
    batch_size_segsr: int,
    lr_segsr: float,
    epochs: int,
    enable_distillation: bool,
    lambda_l1: float,
    lambda_cosine: float,
    lambda_structure: float,
    fold: int,
    **kwargs
):
    if not Path(data_path).exists():
        raise ValueError("Input image path does not exist.")
    if not Path(seg_path).exists():
        raise ValueError("Segmentation results from nnUNet does not exist.")
    os.makedirs(checkpoint_path, exist_ok=True)
    data_merged_sr_h5_path = os.path.join(tmp_path, "data_merged_sr_h5")
    flavr_output_path = os.path.join(tmp_path, "flavr_output")
    data_merged_segsr_h5_path = os.path.join(tmp_path, "data_merged_segsr_h5")
    os.makedirs(data_merged_sr_h5_path, exist_ok=True)
    os.makedirs(flavr_output_path, exist_ok=True)
    os.makedirs(data_merged_segsr_h5_path, exist_ok=True)

    smore_checkpoint_path = os.path.join(checkpoint_path, 'smore')
    flavr_checkpoint_path = os.path.join(checkpoint_path, 'flavr')
    segsr_checkpoint_path = os.path.join(checkpoint_path, 'segsr')
    os.makedirs(smore_checkpoint_path, exist_ok=True)
    os.makedirs(flavr_checkpoint_path, exist_ok=True)
    os.makedirs(segsr_checkpoint_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # preprocess the data
    text_div = "=" * 20
    print(f"{text_div} PROCESSING DATA {text_div}")
    merge_data_path = os.path.join(tmp_path, "data_merged")
    os.makedirs(merge_data_path, exist_ok=True)
    merge_images_and_labels(data_path, merge_data_path)

    # Stage 1 training
    print(f"{text_div} BEGIN TRAINING STAGE ONE {text_div}")
    if fold is None:
        split_data = os.listdir(data_path)
    else:
        split_path = os.path.join(os.pardir(seg_path).replace('nnUNet_results', 'nnUNet_preprocessed'), 'splits_final.json')
        with open(split_path, 'r') as f:
            split_data = json.load(f)[fold]['train']
    loss_obj = torch.nn.L1Loss().to(device)
    loss_seg = BCEDiceLoss(alpha=1, beta=1).to(device)
    slice_separation = float(slice_thickness / target_thickness)
    if smore_initialization:
        n_steps = int(ceil(n_patches / batch_size_sr))
        lr_patch_size = [patch_size, patch_size]
        model = WDSR(
            out_channel=2,
            n_resblocks=16,
            num_channels=32,
            scale=slice_separation,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=lr_sr)
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=opt,
            max_lr=lr_sr,
            total_steps=n_steps,
            cycle_momentum=True,
        )
        patch_size = model.calc_out_patch_size(lr_patch_size)
        train_dataset = TrainSetMultiple(merge_data_path, split_data, slice_thickness, target_thickness, 
                                     None, blur_kernel, patch_size, random_flip, 
                                     device, preload=True, blur=True, nnunet_transform=False)
        data_loader = DataLoader(
            train_dataset,
            batch_size=batch_size_sr,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=4,
        )
        
        opt.step()  # necessary for the LR scheduler
        print(f"\n{text_div} TRAINING NETWORK SMORE {text_div}\n")
        train_sr(n_patches, batch_size_sr, model, opt, scheduler, data_loader, device, 
                 loss_obj, loss_seg, n_steps, slice_separation, 1, False, 
                 smore_checkpoint_path, save_iters_sr)
        model.eval()
        smore_output_path = os.path.join(tmp_path, "smore_output")
        print(f"\n{text_div} INFERENCE NETWORK SMORE {text_div}\n")
        for subject in tqdm(os.listdir(merge_data_path)):
            inference_smore(model, 'img+seg', os.path.join(merge_data_path, subject), os.path.join(data_path, subject), os.path.join(smore_output_path, subject),
                            slice_thickness, target_thickness, device)
            img_hr, label_hr, image_x_rgb, image_y_rgb = postprocess_smore(subject, slice_separation, None, smore_output_path)
            with h5py.File(os.path.join(data_merged_sr_h5_path, subject + '.h5'), 'w') as f:
                f.create_dataset('img_hr', data=img_hr)
                f.create_dataset('label_hr', data=label_hr)
                f.create_dataset('image_x_rgb', data=image_x_rgb)
                f.create_dataset('image_y_rgb', data=image_y_rgb)
    else:
        for subject in tqdm(os.listdir(merge_data_path)):
            img_hr, label_hr, image_x_rgb, image_y_rgb = postprocess_smore(subject, slice_separation, merge_data_path, None)
            with h5py.File(os.path.join(data_merged_sr_h5_path, subject + '.h5'), 'w') as f:
                f.create_dataset('img_hr', data=img_hr)
                f.create_dataset('label_hr', data=label_hr)
                f.create_dataset('image_x_rgb', data=image_x_rgb)
                f.create_dataset('image_y_rgb', data=image_y_rgb)
    
    
    n_steps = int(ceil(n_patches / batch_size_sr))
    slice_separation = float(slice_thickness / target_thickness)
    lr_patch_size = [num_slices, patch_size, patch_size]
    model = UNet_3D_3D(
        img_channels=2,
        block="unet_18",
        n_inputs=num_slices,
        n_outputs=int(slice_separation),
        batchnorm=False,
        joinType="concat",
        upmode="transpose",
        use_uncertainty=False
    ).to(device)

    if pretrain_path is not None:
        pretrained_dict = torch.load(pretrain_path)
        pretrained_dict = pretrained_dict['state_dict']
        for key in list(pretrained_dict.keys()):
            if "module" in key:
                pretrained_dict[key.replace("module.", "")] = pretrained_dict.pop(key)
            if 'encoder.stem.0' in key or 'outconv.1' in key or 'feature_fuse' in key:
                pretrained_dict.pop(key.replace("module.", ""))
        
        model.load_state_dict(pretrained_dict, strict=False)

    opt = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=lr_sr)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=opt,
        max_lr=lr_sr,
        total_steps=n_steps,
        cycle_momentum=True,
    )
    patch_size = model.calc_out_patch_size(lr_patch_size)
    train_dataset = TrainSetMultiple(data_merged_sr_h5_path, split_data, slice_thickness, target_thickness, 
                                    None, blur_kernel, patch_size, random_flip, 
                                    device, preload=True, blur=True, nnunet_transform=nnunet_transform)
    data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_sr,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=0,
    )

    opt.step()  # necessary for the LR scheduler
    print(f"\n{text_div} TRAINING NETWORK FLAVR {text_div}\n")
    train_sr(n_patches, batch_size_sr, model, opt, scheduler, data_loader, device, 
                loss_obj, loss_seg, n_steps, slice_separation, num_slices, False, 
                flavr_checkpoint_path, save_iters_sr)
    if enable_uncertainty:
        state_dict = model.state_dict()
        model = UNet_3D_3D(
            img_channels=2,
            block="unet_18",
            n_inputs=num_slices,
            n_outputs=int(slice_separation),
            batchnorm=False,
            joinType="concat",
            upmode="transpose",
            use_uncertainty=True
        ).to(device)

        pretrained_dict = state_dict
        for key in list(pretrained_dict.keys()):
            if "module" in key:
                pretrained_dict[key.replace("module.", "")] = pretrained_dict.pop(key)
            if 'outconv.1' in key or 'feature_fuse' in key:
                pretrained_dict.pop(key.replace("module.", ""))

        model.load_state_dict(pretrained_dict, strict=False)
        opt = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=lr_sr)
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=opt,
            max_lr=lr_sr,
            total_steps=n_steps,
            cycle_momentum=True,
        )

        opt.step()  # necessary for the LR scheduler
        print(f"\n{text_div} TRAINING NETWORK FLAVR WITH UNCERTAINTY {text_div}\n")
        train_sr(n_patches, batch_size_sr, model, opt, scheduler, data_loader, device, 
                    loss_obj, loss_seg, 20000, slice_separation, num_slices, True, 
                    flavr_checkpoint_path, save_iters_sr)
    print(f"\n{text_div} INFERENCE NETWORK FLAVR {text_div}\n")
    model.eval()
    for subject in tqdm(os.listdir(merge_data_path)):
        inference_flavr(model, 'img+seg', os.path.join(merge_data_path, subject), os.path.join(data_path, subject), os.path.join(flavr_output_path, subject),
                        slice_thickness, target_thickness, device, True)
        image, seg, uncertainty = postprocess_smore(subject, slice_separation, flavr_output_path)
        with h5py.File(os.path.join(data_merged_segsr_h5_path, subject + '.h5'), 'w') as f:
            f.create_dataset('img', data=image)
            f.create_dataset('seg', data=seg)
            f.create_dataset('uncertainty', data=uncertainty)

    # Stage 2 training
    print(f"\n{text_div} TRAINING NETWORK REHRSeg {text_div}\n")
    plan_file = os.path.join(seg_path, 'plans.json')
    with open(plan_file, 'r') as f:
        plan = json.load(f)
    patch_size = plan['configurations']['3d_fullres']['patch_size'][::-1]
    patch_size_ori = [patch_size[0]+64, patch_size[1]+64, patch_size[2]]
    enable_deep_supervision=False
    model_sr = model
    arch_kwargs = plan['configurations']['3d_fullres']['architecture']['arch_kwargs']
    model_seg = SegModel(
        input_channels=1,
        num_classes=2,
        n_stages=arch_kwargs['n_stages'],
        upscale=int(slice_separation),
        features_per_stage=arch_kwargs['features_per_stage'],
        conv_op=torch.nn.modules.conv.Conv3d,
        kernel_sizes=arch_kwargs['kernel_sizes'],
        strides=arch_kwargs['strides'],
        n_conv_per_stage=arch_kwargs['n_conv_per_stage'],
        n_conv_per_stage_decoder=arch_kwargs['n_conv_per_stage_decoder'], 
        conv_bias=arch_kwargs['conv_bias'],
        norm_op=torch.nn.modules.instancenorm.InstanceNorm3d,
        norm_op_kwargs=arch_kwargs['norm_op_kwargs'],
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs=arch_kwargs['nonlin_kwargs'],
        deep_supervision=enable_deep_supervision
    ).to(device)
    if enable_distillation:
        distiller = Distiller(64, 64, lambda_l1, lambda_cosine, lambda_structure).to(device)
    resume_seg_path = os.path.join(seg_path, f'fold_{fold}', 'checkpoint_final.pth')
    checkpoint_seg = torch.load(resume_seg_path, map_location=torch.device('cpu'))
    parameter_seg = checkpoint_seg['network_weights'] if 'network_weights' in checkpoint_seg else checkpoint_seg['model']
    model_seg.load_state_dict(parameter_seg, strict=False)
    train_dataset = TrainSetMultipleSegSREfficient(data_merged_segsr_h5_path, split_data, slice_thickness, target_thickness, patch_size_ori, patch_size, random_flip, enable_uncertainty)
    data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_segsr,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
    )
    separated_params = separate_weight_extensive_params(model_seg, False, lr_segsr)
    if enable_distillation:
        separated_params = itertools.chain(model_seg.parameters(), distiller.parameters())
        # separated_params[0]['params'] = list(separated_params[0]['params']) + list(distiller_1.parameters()) + list(distiller_2.parameters())
    opt = torch.optim.SGD(separated_params, lr=lr_segsr, momentum=0.99, nesterov=True, weight_decay = 3e-5)
    lr_scheduler = PolynomialLR(opt, total_iters=epochs)

    loss_obj_lr_seg = _build_loss(enable_deep_supervision=enable_deep_supervision, weight_dice=0 if enable_uncertainty else 1)
    loss_obj_hr_seg = _build_loss(enable_deep_supervision=False, weight_dice=1)

    total_iters = 0
    for _ in range(epochs):
        for img, label_lr, label, uncertainty_lr in data_loader:
            model_seg.train()
            
            pseudo_img_lr = img.to(device)
            if enable_deep_supervision:
                pseudo_label_lr = [x.to(device) for x in label_lr]
            else:
                pseudo_label_lr = label_lr.to(device)
            label_sr = label.to(device)
            
            if enable_distillation:
                with torch.no_grad():
                    features_sr = get_intermediate_features(model_sr, pseudo_img_lr, pseudo_label_lr, device)
                pseudo_seg_lr, seg_sr, features_seg = model_seg(pseudo_img_lr, return_inetermediate_feature=True)
            else:
                pseudo_seg_lr, seg_sr = model_seg(pseudo_img_lr)

            if enable_uncertainty:
                pseudo_uncertainty_lr = uncertainty_lr.to(device)

                loss_lr_seg = loss_obj_lr_seg(pseudo_seg_lr, pseudo_label_lr, pseudo_uncertainty_lr)
                loss_hr_seg = loss_obj_hr_seg(seg_sr, label_sr, None)
            else:
                loss_lr_seg = loss_obj_lr_seg(pseudo_seg_lr, pseudo_label_lr)
                loss_hr_seg = loss_obj_hr_seg(seg_sr, label_sr)

            loss = loss_lr_seg + loss_hr_seg
            if enable_distillation:
                distill_loss = 0
                distill_loss += distiller(features_seg[1], features_sr[1])

                loss += distill_loss
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_iters += 1

            if total_iters % save_iters_segsr == 0:
                model_seg.eval()
                model_seg.decoder.deep_supervision = False
                val_dice = evaluate(model_seg, patch_size_ori, data_path, data_path.replace('imagesTr', 'labelsTr'), split_path, fold, save_path=None, eval_HR=False, seperation=slice_separation)
                print(f"Eval result: {val_dice}")
                print(f"Current lr:", opt.param_groups[0]['lr'])
                torch.save(
                    {
                        "model": model_seg.state_dict(),
                        "optimizer": opt.state_dict(),
                        "scheduler": lr_scheduler.state_dict(),
                    },
                    segsr_checkpoint_path / f"weights_{total_iters}_{val_dice}.pt",
                )
                model_seg.decoder.deep_supervision = enable_deep_supervision
        lr_scheduler.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/brain.yaml")
    parser.add_argument("--fold", type=int, default=None)
    args = parser.parse_args()

    main(**OmegaConf.load(args.config), fold=args.fold)
