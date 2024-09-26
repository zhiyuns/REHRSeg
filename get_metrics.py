import os
import sys
import json
import argparse
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.spatial.distance import directed_hausdorff
from medpy import metric

def resample_img(itk_image, out_spacing=[0.6, 0.6, 3.0], out_size=[], is_label=True):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

class ConfusionMatrix:

    def __init__(self, test=None, reference=None):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.test_empty = None
        self.test_full = None
        self.set_reference(reference)
        self.set_test(test)

    def set_test(self, test):

        self.test = test
        self.reset()

    def set_reference(self, reference):

        self.reference = reference
        self.reset()

    def reset(self):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.test_empty = None
        self.test_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):

        if self.test is None or self.reference is None:
            raise ValueError("'test' and 'reference' must both be set to compute confusion matrix.")

        self.tp = int(((self.test != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.test != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.test == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.test == 0) * (self.reference != 0)).sum())
        self.size = int(np.prod(self.reference.shape, dtype=np.int64))
        self.test_empty = not np.any(self.test)
        self.test_full = np.all(self.test)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):

        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):

        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):

        for case in (self.test_empty, self.test_full, self.reference_empty, self.reference_full):
            if case is None:
                self.compute()
                break

        return self.test_empty, self.test_full, self.reference_empty, self.reference_full


def Dice(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=False, **kwargs):
    """2TP / (2TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(2. * tp / (2 * tp + fp + fn))

def convert_corner_to_true(segmentation):
    a, b, c = segmentation.shape
    segmentation[0, 0, 0] = True
    segmentation[a - 1, 0, 0] = True
    segmentation[0, b - 1, 0] = True
    segmentation[0, 0, c - 1] = True

    segmentation[a - 1, b - 1, 0] = True
    segmentation[a - 1, b - 1, c - 1] = True
    segmentation[0, b - 1, c - 1] = True
    segmentation[a - 1, 0, c - 1] = True
    return segmentation

def HausdorffDistance95(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=False, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return HausdorffDistance(test=test, reference=reference, confusion_matrix=confusion_matrix, nan_for_nonexisting=nan_for_nonexisting, voxel_spacing=voxel_spacing, connectivity=connectivity)

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.hd95(test, reference, voxel_spacing, connectivity)

def HausdorffDistance(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=False, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:


        if nan_for_nonexisting:
            return float("NaN")
        else:
            confusion_matrix.test = convert_corner_to_true(confusion_matrix.test)

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.hd(test, reference, voxel_spacing, connectivity)

def calculate_metrics(prediction, ground_truth, spacing):
    # dice
    dice = Dice(test=prediction, reference=ground_truth)
    # hausdorff 95
    hausdorff95 = HausdorffDistance95(test=prediction, reference=ground_truth, voxel_spacing=spacing)
    # hausdorff
    hausdorff = HausdorffDistance(test=prediction, reference=ground_truth, voxel_spacing=spacing)

    return dice, hausdorff95, hausdorff


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-label-path-hr", type=str, default=r'../../../data/Brain_datasets/Meningioma_converted_preprocessed')
    parser.add_argument("--val-label-path-lr", type=str, default=r'../../../data/Brain_datasets/Meningioma_converted_preprocessed_lr')
    parser.add_argument("--val-pred-path", type=str, default=None)
    parser.add_argument("--split-path", type=str, default='../../nnUNet/DATASET/nnUNet_preprocessed/Dataset509_BoneTumor/splits_final.json')
    parser.add_argument("--plan-file", type=str, default='../../nnUNet/DATASET/nnUNet_results/Dataset509_BoneTumor/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json')
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--eval-LR", action="store_true", default=False)
    parser.add_argument("--eval-HR", action="store_true", default=False)

    args = parser.parse_args(args if args is not None else sys.argv[1:])
    val_label_path_hr = args.val_label_path_hr
    val_label_path_lr = args.val_label_path_lr
    val_pred_path = args.val_pred_path
    eval_LR = args.eval_LR
    eval_HR = args.eval_HR
    with open(args.plan_file, 'r') as f:
        plan = json.load(f)
    if args.fold is not None:
        with open(args.split_path, 'r') as f:
            split_data = json.load(f)[args.fold]['val']
    else:
        split_data = os.listdir(val_label_path_hr)

    bad_case = ['079', '011', '086', '072', '056']
    bad_case_inhouse = ['98', '96', '89', '76', '71', '79', '43', '58', '42', '20', '123', '35', '']
    
    all_dice_lr = []
    all_hausdorff95_lr = []
    all_hausdorff_lr = []
    all_dice_hr = []
    all_hausdorff95_hr = []
    all_hausdorff_hr = []

    text_all = []

    for subject in split_data:
        if subject.split('-')[-1] in bad_case or subject.split('-')[0] in bad_case_inhouse:
            continue
        if eval_LR:
            if os.path.exists(os.path.join(val_label_path_lr, subject, 'label.nii.gz')):
                each_label_lr = os.path.join(val_label_path_lr, subject, 'label.nii.gz')
            else:
                each_label_lr = os.path.join(val_label_path_lr, subject+'.nii.gz')
            label_lr = nib.load(each_label_lr).get_fdata()
            if os.path.exists(os.path.join(val_pred_path, subject+'_pred_lr.nii.gz')):
                each_pred_lr = os.path.join(val_pred_path, subject+'_pred_lr.nii.gz')
            elif os.path.exists(os.path.join(val_pred_path, subject+'_pred.nii.gz')):
                each_pred_lr = os.path.join(val_pred_path, subject+'_pred.nii.gz')
            else:
                each_pred_lr = os.path.join(val_pred_path, subject+'.nii.gz')
                
            pred_lr = nib.load(each_pred_lr).get_fdata()

            voxel_spacing_lr = np.array(sitk.ReadImage(each_label_lr).GetSpacing())[::-1]
            dice_lr, hausdorff95_lr, hausdorff_lr = calculate_metrics(pred_lr, label_lr, voxel_spacing_lr)
        else:
            dice_lr = 0
            hausdorff95_lr = 0
            hausdorff_lr = 0

        if eval_HR:
            each_label_hr = os.path.join(val_label_path_hr, subject, 'label.nii.gz')
            label_hr = nib.load(each_label_hr).get_fdata()
            if os.path.exists(os.path.join(val_pred_path, subject+'_pred_hr.nii.gz')):
                each_pred_hr = os.path.join(val_pred_path, subject+'_pred_hr.nii.gz')
            elif os.path.exists(os.path.join(val_pred_path, subject+'_seg.nii.gz')):
                each_pred_hr = os.path.join(val_pred_path, subject+'_seg.nii.gz')
            elif os.path.exists(os.path.join(val_pred_path, subject+'_pred.nii.gz')):
                each_pred_hr = os.path.join(val_pred_path, subject+'_pred.nii.gz')
            else:
                each_pred_hr = os.path.join(val_pred_path, subject+'.nii.gz')
            pred_hr = nib.load(each_pred_hr).get_fdata()

            # check shape
            if label_hr.shape != pred_hr.shape:
                print(f'Error: {subject} hr shape mismatch, resampling from {pred_hr.shape} to {label_hr.shape}')
                label_hr_spacing = sitk.ReadImage(each_label_hr).GetSpacing()
                label_hr_size = sitk.ReadImage(each_label_hr).GetSize()
                pred_hr_sitk = sitk.ReadImage(each_pred_hr)
                if label_hr.shape[2] + 4 % pred_hr.shape[2] == 0:
                    pred_hr_data = sitk.GetArrayFromImage(pred_hr_sitk)
                    pred_hr_data = np.pad(pred_hr_data, ((0, 0), (0, 0), (0, 4)), 'constant', constant_values=0)
                    pred_hr_sitk_out = sitk.GetImageFromArray(pred_hr_data)
                    pred_hr_sitk_out.CopyInformation(pred_hr_sitk)
                    pred_hr_sitk = pred_hr_sitk_out
                elif label_hr.shape[2] - 4 % pred_hr.shape[2] == 0:
                    pred_hr_data = sitk.GetArrayFromImage(pred_hr_sitk)
                    pred_hr_data = pred_hr_data[:, :, 0:label_hr.shape[2]]
                    pred_hr_sitk_out = sitk.GetImageFromArray(pred_hr_data)
                    pred_hr_sitk_out.CopyInformation(pred_hr_sitk)
                    pred_hr_sitk = pred_hr_sitk_out

                pred_hr_sitk = resample_img(pred_hr_sitk, out_spacing=label_hr_spacing, out_size=label_hr_size, is_label=True)
                sitk.WriteImage(pred_hr_sitk, each_pred_hr)
                pred_hr = nib.load(each_pred_hr).get_fdata()

            voxel_spacing_hr = np.array(sitk.ReadImage(each_label_hr).GetSpacing())[::-1]
            dice_hr, hausdorff95_hr, hausdorff_hr = calculate_metrics(pred_hr, label_hr, voxel_spacing_hr)
        else:
            dice_hr = 0
            hausdorff95_hr = 0
            hausdorff_hr = 0
        
        # judge nan
        if np.isnan(hausdorff_lr) or np.isnan(hausdorff_hr):
            print(f'Error: {subject} hausdorff distance is nan')
            continue
        
        all_dice_lr.append(dice_lr)
        all_hausdorff95_lr.append(hausdorff95_lr)
        all_hausdorff_lr.append(hausdorff_lr)
        all_dice_hr.append(dice_hr)
        all_hausdorff95_hr.append(hausdorff95_hr)
        all_hausdorff_hr.append(hausdorff_hr)

        print(f'Subject: {subject}, Dice LR: {dice_lr}, Hausdorff95 LR: {hausdorff95_lr}, Hausdorff LR: {hausdorff_lr}, Dice HR: {dice_hr}, Hausdorff95 HR: {hausdorff95_hr}, Hausdorff HR: {hausdorff_hr}')
        text_all.append(f'Subject: {subject}, Dice LR: {dice_lr}, Hausdorff95 LR: {hausdorff95_lr}, Hausdorff LR: {hausdorff_lr}, Dice HR: {dice_hr}, Hausdorff95 HR: {hausdorff95_hr}, Hausdorff HR: {hausdorff_hr}')

    print('Dice LR:', np.mean(all_dice_lr))
    print('Hausdorff95 LR:', np.mean(all_hausdorff95_lr))
    print('Hausdorff LR:', np.mean(all_hausdorff_lr))
    print('Dice HR:', np.mean(all_dice_hr))
    print('Hausdorff95 HR:', np.mean(all_hausdorff95_hr))
    print('Hausdorff HR:', np.mean(all_hausdorff_hr))
    print('Dice LR std:', np.std(all_dice_lr))
    print('Hausdorff95 LR std:', np.std(all_hausdorff95_lr))
    print('Hausdorff LR std:', np.std(all_hausdorff_lr))
    print('Dice HR std:', np.std(all_dice_hr))
    print('Hausdorff95 HR std:', np.std(all_hausdorff95_hr))
    print('Hausdorff HR std:', np.std(all_hausdorff_hr))
    text_all.append(f'Dice LR: {np.mean(all_dice_lr)}')
    text_all.append(f'Hausdorff95 LR: {np.mean(all_hausdorff95_lr)}')
    text_all.append(f'Hausdorff LR: {np.mean(all_hausdorff_lr)}')
    text_all.append(f'Dice HR: {np.mean(all_dice_hr)}')
    text_all.append(f'Hausdorff95 HR: {np.mean(all_hausdorff95_hr)}')
    text_all.append(f'Hausdorff HR: {np.mean(all_hausdorff_hr)}')
    text_all.append(f'Dice LR std: {np.std(all_dice_lr)}')
    text_all.append(f'Hausdorff95 LR std: {np.std(all_hausdorff95_lr)}')
    text_all.append(f'Hausdorff LR std: {np.std(all_hausdorff_lr)}')
    text_all.append(f'Dice HR std: {np.std(all_dice_hr)}')
    text_all.append(f'Hausdorff95 HR std: {np.std(all_hausdorff95_hr)}')
    text_all.append(f'Hausdorff HR std: {np.std(all_hausdorff_hr)}')
    with open(os.path.join(val_pred_path, 'metrics.txt'), 'w') as f:
        for item in text_all:
            f.write("%s\n" % item)

if __name__ == "__main__":
    main()




