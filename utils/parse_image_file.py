import h5py
import numpy as np
import nibabel as nib
from degrade.degrade import fwhm_units_to_voxel_space, fwhm_needed


def normalize(x, a=-1, b=1):
    orig_min = x.min()
    orig_max = x.max()

    numer = (x - orig_min) * (b - a)
    denom = orig_max - orig_min

    return a + numer / denom, orig_min, orig_max


def inv_normalize(x, orig_min, orig_max, a=-1, b=1):
    tmp = x - a
    tmp = tmp * (orig_max - orig_min)
    tmp = tmp / (b - a)
    tmp += orig_min
    return tmp

class LazyHDF5File:
    """Implementation of the LazyHDF5File class for the LazyHDF5Dataset."""

    def __init__(self, path, internal_path=None):
        self.path = path
        self.internal_path = internal_path
        if self.internal_path:
            with h5py.File(self.path, "r") as f:
                self.ndim = f[self.internal_path].ndim
                self.shape = f[self.internal_path].shape
    
    def ravel(self):
        with h5py.File(self.path, "r") as f:
            data = f[self.internal_path][:].ravel()
        return data

    def __getitem__(self, arg):
        if isinstance(arg, str) and not self.internal_path:
            return LazyHDF5File(self.path, arg)

        if arg == Ellipsis:
            return LazyHDF5File(self.path, self.internal_path)

        with h5py.File(self.path, "r") as f:
            data = f[self.internal_path][arg]

        return data

def parse_image(img_file, slice_thickness=None, target_thickness=None):
    """
    打开图像体积文件，并返回相关信息：
    - 图像数组作为一个numpy数组
    - 各向异性的“尺度”（即切片间隔）
    - 低分辨率轴
    - PSF（切片厚度）的FWHM（可以作为参数提供）
    - The header of the image file
    - The affine matrix of the image file
    """
    if img_file.endswith(".nii.gz"):
        obj = nib.load(img_file)
        voxel_size = tuple(float(v) for v in obj.header.get_zooms())
        image = obj.get_fdata(dtype=np.float32)

        # x, y, and z are the spatial physical measurement sizes
        lr_axis = np.argmax(voxel_size) if len(voxel_size) == 2 else 0

        image = obj.get_fdata(dtype=np.float32)  # 保持原始维度
        header = obj.header
        affine = obj.affine
        orig_min = image.min()
        orig_max = image.max()
    elif img_file.endswith(".h5"):
        image = LazyHDF5File(img_file)
        header = None
        affine = None
        lr_axis = None
        orig_min = None
        orig_max = None
    slice_separation = float(slice_thickness / target_thickness)

    # 计算模糊核的FWHM（以体素为单位）
    blur_fwhm_voxels = fwhm_units_to_voxel_space(fwhm_needed(target_thickness, slice_thickness), target_thickness)

    return (
        image,
        slice_separation,
        lr_axis,  # 低分辨率轴的占位符，因为它不是这个函数明确返回的
        blur_fwhm_voxels,
        header,
        affine,
        orig_min,
        orig_max,
    )



def lr_axis_to_z(img, lr_axis):
    """
    Orient the image volume such that the low-resolution axis
    is in the "z" axis.
    """
    print("img", img.shape)
    if img.ndim == 5:
        # img = np.squeeze(img, axis=4)
        img = np.squeeze(img)

    if lr_axis == 0:
        return img.transpose(2, 0, 1, 3)
    elif lr_axis == 1:
        return img.transpose(1, 2, 0, 3)
    elif lr_axis == 2:
        return img



def z_axis_to_lr_axis(img, lr_axis):
    """
    Orient the image volume such that the "z" axis
    is back to the original low-resolution axis
    """
    if img.ndim == 5:
        img = np.squeeze(img, axis=4)

    if lr_axis == 0:
        return img.transpose(2, 0, 1, 3)
    elif lr_axis == 1:
        return img.transpose(1, 2, 0, 3)
    elif lr_axis == 2:
        return img
