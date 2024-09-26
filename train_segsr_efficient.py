import argparse
import os
import sys
from pathlib import Path
import time
import itertools
import torch
import torch.nn.functional as F
import SimpleITK as sitk
from torch.utils.data import DataLoader
from math import ceil
import random
import numpy as np
import json
import itertools

from models.FLAVR.FLAVR_arch import UNet_3D_3D

from models.seg_model import SegModel, Distiller

from utils.train_set import TrainSetMultipleSegSREfficient
from utils.misc_utils import parse_device
from utils.seg_utils import zscore_normalization, zeroone_normalization, percentile_normalization, _build_loss, calculate_dice, evaluate_case, get_training_transforms

# from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from torch.optim.lr_scheduler import PolynomialLR

# from thop import profile

def get_batch_slice_from_idx(img, start_idx):
    out_img = []
    for i in range(img.shape[0]):
        out_img.append(img[i:i+1,:,start_idx[i]:start_idx[i]+1,...])
    out_img = torch.cat(out_img, dim=0)
    return out_img

def pad_to_target(img, target_size):
    if target_size > img.shape[2]:
        pad_size = [0,0,0,0,0,target_size-img.shape[2]]
        out = torch.nn.functional.pad(img, pad_size)
    else:
        out = img[:,:,:target_size,...]
    return out

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

def evaluate(model_seg, patch_size_ori, test_img_path, test_label_path, val_img_path, val_label_path, split_path, fold, val_result=True, test_result=False, save_path=None, eval_HR=False, seperation=1):
    # test
    if test_result and test_img_path is not None:
        all_dice = []
        all_pred = []
        all_label = []
        for subject in os.listdir(test_img_path):
            subject = subject.split('_')[0]
            test_img = os.path.join(test_img_path, subject+'_0000.nii.gz')
            test_label = os.path.join(test_label_path, subject+'.nii.gz')
            pred_lr, pred_hr, label_lr, dice_lr = evaluate_case(model_seg, test_img, test_label, slice_separation=seperation, patch_size=patch_size_ori[::-1], get_HR_results=eval_HR)
            if save_path is not None:
                os.makedirs(os.path.join(save_path, 'test'), exist_ok=True)
                ori_info = sitk.ReadImage(test_img)
                #copy information
                pred_lr_img = sitk.GetImageFromArray(pred_lr)
                pred_lr_img.SetSpacing(ori_info.GetSpacing())
                pred_lr_img.SetOrigin(ori_info.GetOrigin())
                pred_lr_img.SetDirection(ori_info.GetDirection())
                sitk.WriteImage(pred_lr_img, os.path.join(save_path, 'test', f"{subject}_pred_lr.nii.gz"))

                if eval_HR:
                    pred_hr_img = sitk.GetImageFromArray(pred_hr)
                    ori_spacing = ori_info.GetSpacing()
                    pred_hr_img.SetSpacing([ori_spacing[0], ori_spacing[1], ori_spacing[2]/seperation])
                    pred_hr_img.SetOrigin(ori_info.GetOrigin())
                    pred_hr_img.SetDirection(ori_info.GetDirection())
                    sitk.WriteImage(pred_hr_img, os.path.join(save_path, 'test', f"{subject}_pred_hr.nii.gz"))
            all_pred.append(pred_lr.flatten())
            all_label.append(label_lr.flatten())
            print(f"Subject {subject}: {dice_lr}")
            all_dice.append(dice_lr)
        print(f'Global dice: {calculate_dice(np.concatenate(all_pred), np.concatenate(all_label))}')
        print(f"Average dice: {sum(all_dice)/len(all_dice)}")
        print(f"Std dice: {np.std(all_dice)}")
        print(f"Max dice: {max(all_dice)}")
        print(f"Min dice: {min(all_dice)}")
        test_dice = sum(all_dice)/len(all_dice)
    else:
        test_dice = 0
    
    if val_result and val_img_path is not None:
        # val
        # bad_cases = ['44-dicom', '54-dicom', '75-dicom', '94-dicom', '97-dicom', 
        #                 '19-t1c', '30-t1c', '32-t1c', '15-t1c', '16-t1c', '55-dicom',]
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
    else:
        val_dice = 0
        
    return test_dice, val_dice

def main(args=None):
    #################### ARGUMENTS ####################
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-fpath", type=str, required=True)
    parser.add_argument("--weight-dir", type=str, required=True)
    parser.add_argument("--gpu-id", type=int, default=-1)
    parser.add_argument("--interp-order", type=int, default=3)
    parser.add_argument("--print-iters", type=int, default=1000)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument(
        "--save-iters", type=int, default=10000
    )  # The sum of 4 phases from Shuo's thesis
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-slices", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--num-blocks", type=int, default=16)
    parser.add_argument("--num-channels", type=int, default=32)
    parser.add_argument("--slice-thickness", type=float)
    parser.add_argument("--target-thickness", type=float)
    parser.add_argument("--patch-sampling", type=str, default="gradient")
    parser.add_argument("--resume-sr-path", type=str, default=None)
    parser.add_argument("--resume-seg-path", type=str, default=None)
    parser.add_argument("--plan-file", type=str, default='../../nnUNet/DATASET/nnUNet_results/Dataset506_BoneTumor/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json')
    parser.add_argument("--split-path", type=str, default='../../nnUNet/DATASET/nnUNet_preprocessed/Dataset506_BoneTumor/splits_final_new.json')
    parser.add_argument("--val-img-path", type=str, default=None) # '../../nnUNet/DATASET/nnUNet_raw/Dataset506_BoneTumor/imagesTr'
    parser.add_argument("--val-label-path", type=str, default=None) # '../../nnUNet/DATASET/nnUNet_raw/Dataset506_BoneTumor/labelsTr_new'
    parser.add_argument("--test-img-path", type=str, default=None) # r'../../../data/preprocess/test_img'
    parser.add_argument("--test-label-path", type=str, default=None) # r'../../../data/preprocess/test_label'
    parser.add_argument("--enable-uncertainty", action="store_true", default=False)
    parser.add_argument("--enable-distillation", action="store_true", default=False)
    parser.add_argument("--lambda_l1", type=float, default=0.0)
    parser.add_argument("--lambda_cosine", type=float, default=0.0)
    parser.add_argument("--lambda_structure", type=float, default=0.0)
    parser.add_argument("--manual-seed", type=int, default=None)
    parser.add_argument("--use-diceloss", action="store_true", default=False)
    parser.add_argument("--random-flip", action="store_true", default=False)
    parser.add_argument("--eval-only", action="store_true", default=False)
    parser.add_argument("--no-eval-HR", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)

    args = parser.parse_args(args if args is not None else sys.argv[1:])

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True

    test_img = None # r'../../../data/version2/83/83-dicom.nii.gz'
    test_label = None # r'../../../data/version2/83/83-seg.nii.gz'
    split_path = args.split_path
    val_img_path = args.val_img_path
    val_label_path = args.val_label_path
    test_img_path = args.test_img_path
    test_label_path = args.test_label_path
    
    
    # A nice print statement divider for the command line
    text_div = "=" * 10

    print(f"{text_div} BEGIN TRAINING {text_div}")

    
    weight_dir = Path(args.weight_dir)
    learning_rate = args.lr
    device = parse_device(args.gpu_id)

    if not weight_dir.exists():
        weight_dir.mkdir(parents=True)

    plan_file = args.plan_file
    with open(plan_file, 'r') as f:
        plan = json.load(f)
    patch_size = plan['configurations']['3d_fullres']['patch_size'][::-1]

    slice_separation = float(args.slice_thickness / args.target_thickness)
    patch_size_ori = [patch_size[0]+64, patch_size[1]+64, patch_size[2]] # [448,448,16]

    enable_deep_supervision=False
    
    # ===== MODEL SETUP =====

    if args.resume_sr_path == 'trilinear':
        model_sr = None
    else:
        model_sr = UNet_3D_3D(
            img_channels=2,
            block="unet_18",
            n_inputs=args.num_slices,
            n_outputs=int(slice_separation),
            batchnorm=False,
            joinType="concat",
            upmode="transpose",
            use_uncertainty=True
        ).eval().to(device)

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

    # profile_input = torch.randn(1, 1, 32, 128, 128).to(device)
    # flops, params = profile(model_seg, inputs=(profile_input,))
    # print(f"GFLOPs: {flops / 1e9}, Params: {params / 1e6}")

    if args.enable_distillation:
        distiller_1 = Distiller(64, 64, args.lambda_l1, args.lambda_cosine, args.lambda_structure).to(device)
        # distiller_2 = Distiller(128, 128, args.lambda_l1, args.lambda_cosine, args.lambda_structure).to(device)

    if model_sr is not None and args.resume_sr_path is not None:
        model_sr.load_state_dict(torch.load(args.resume_sr_path)["model"])

    if args.resume_seg_path is not None:
        checkpoint_seg = torch.load(args.resume_seg_path, map_location=torch.device('cpu'))
        parameter_seg = checkpoint_seg['network_weights'] if 'network_weights' in checkpoint_seg else checkpoint_seg['model']
        model_seg.load_state_dict(parameter_seg, strict=False)
    
    model_seg.eval()
    
    if args.eval_only:
        model_seg.eval()
        model_seg.decoder.deep_supervision = False
        save_path = os.path.join(os.path.dirname(args.resume_seg_path), 'evaluate')
        test_dice, val_dice = evaluate(model_seg, patch_size, test_img_path, test_label_path, val_img_path, val_label_path, split_path, args.fold, val_result=True, test_result=True, save_path=save_path, eval_HR=not args.no_eval_HR, seperation=int(slice_separation))
        return

    if args.fold is not None:
        with open(split_path, 'r') as f:
            split_data = json.load(f)[args.fold]['train']
    else:
        split_data = os.listdir(args.in_fpath)
        split_data = [x.split('_0000.h5')[0] for x in split_data]
    train_dataset = TrainSetMultipleSegSREfficient(args.in_fpath, split_data, args.slice_thickness, args.target_thickness, patch_size_ori, patch_size, args.random_flip, args.enable_uncertainty)
    data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
    )

    separated_params = separate_weight_extensive_params(model_seg, False, learning_rate)
    if args.enable_distillation:
        separated_params = itertools.chain(model_seg.parameters(), distiller_1.parameters())
        # separated_params[0]['params'] = list(separated_params[0]['params']) + list(distiller_1.parameters()) + list(distiller_2.parameters())
    opt = torch.optim.SGD(separated_params, lr=learning_rate, momentum=0.99, nesterov=True, weight_decay = 3e-5)
    lr_scheduler = PolynomialLR(opt, total_iters=args.num_epochs)

    loss_obj_lr_seg = _build_loss(enable_deep_supervision=enable_deep_supervision, weight_dice=0 if args.enable_uncertainty else 1)
    loss_obj_hr_seg = _build_loss(enable_deep_supervision=False, weight_dice=1)

    total_iters = 0
    
    for epoch in range(args.num_epochs):
        for i, (img, label_lr, label, uncertainty_lr) in enumerate(data_loader):
            model_seg.train()
            
            pseudo_img_lr = img.to(device)
            if enable_deep_supervision:
                pseudo_label_lr = [x.to(device) for x in label_lr]
            else:
                pseudo_label_lr = label_lr.to(device)
            label_sr = label.to(device)
            
            if args.enable_distillation:
                with torch.no_grad():
                    features_sr = get_intermediate_features(model_sr, pseudo_img_lr, pseudo_label_lr, device)
                pseudo_seg_lr, seg_sr, features_seg = model_seg(pseudo_img_lr, return_inetermediate_feature=True)
            else:
                pseudo_seg_lr, seg_sr = model_seg(pseudo_img_lr)

            if args.enable_uncertainty:
                pseudo_uncertainty_lr = uncertainty_lr.to(device)

                loss_lr_seg = loss_obj_lr_seg(pseudo_seg_lr, pseudo_label_lr, pseudo_uncertainty_lr)
                loss_hr_seg = loss_obj_hr_seg(seg_sr, label_sr, None)
            else:
                loss_lr_seg = loss_obj_lr_seg(pseudo_seg_lr, pseudo_label_lr)
                loss_hr_seg = loss_obj_hr_seg(seg_sr, label_sr)

            loss = loss_lr_seg + loss_hr_seg
            if args.enable_distillation:
                distill_loss = 0
                distill_loss += distiller_1(features_seg[1], features_sr[1])
                # distill_loss += distiller_2(features_seg[2][:,:,2:-2,...], features_sr[2][:,:,2:-2,...])  # only distill the intermediate slices
                # save intermediate visualization
                # if total_iters % 10 == 0:
                #     sitk.WriteImage(sitk.GetImageFromArray(pseudo_img_lr[0,0].detach().cpu().numpy()), f"pseudo_img_{i}.nii.gz")
                #     sitk.WriteImage(sitk.GetImageFromArray(pseudo_label_lr[0,0].detach().cpu().numpy()), f"pseudo_label_{i}.nii.gz")
                #     sitk.WriteImage(sitk.GetImageFromArray(features_seg[1][0,0].detach().cpu().numpy()), f"features_seg_1_{i}.nii.gz")
                #     sitk.WriteImage(sitk.GetImageFromArray(features_sr[1][0,0].detach().cpu().numpy()), f"features_sr_1_{i}.nii.gz")

                loss += distill_loss * 0.1
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_iters += 1

            if total_iters % args.save_iters == 0:
                model_seg.eval()
                model_seg.decoder.deep_supervision = False
                test_dice, val_dice = evaluate(model_seg, patch_size, test_img_path, test_label_path, val_img_path, val_label_path, split_path, args.fold, val_result=True, test_result=True)
                print(f"Eval result: {val_dice}, {test_dice}")
                print(f"Current lr:", opt.param_groups[0]['lr'])
                torch.save(
                    {
                        "model": model_seg.state_dict(),
                        "optimizer": opt.state_dict(),
                        "scheduler": lr_scheduler.state_dict(),
                    },
                    weight_dir / f"weights_{total_iters}_{val_dice}_{test_dice}.pt",
                )
                model_seg.decoder.deep_supervision = enable_deep_supervision
        lr_scheduler.step()

            
if __name__ == "__main__":
    main()