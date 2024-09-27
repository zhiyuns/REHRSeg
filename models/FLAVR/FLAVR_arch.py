import math
import numpy as np
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_3D import SEGating
from resize.pytorch import resize
import SimpleITK as sitk

from einops import rearrange, repeat

def joinTensors(X1 , X2 , type="concat"):

    if type == "concat":
        return torch.cat([X1 , X2] , dim=1)
    elif type == "add":
        return X1 + X2
    else:
        return X1


class Conv_2d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=False, batchnorm=False):

        super().__init__()
        self.conv = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

        if batchnorm:
            self.conv += [nn.BatchNorm2d(out_ch)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):

        return self.conv(x)

class upConv3D(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, upmode="transpose" , batchnorm=False):

        super().__init__()

        self.upmode = upmode

        if self.upmode=="transpose":
            self.upconv = nn.ModuleList(
                [nn.ConvTranspose3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
                SEGating(out_ch)
                ]
            )

        else:
            self.upconv = nn.ModuleList(
                [nn.Upsample(mode='trilinear', scale_factor=(1,2,2), align_corners=False),
                nn.Conv3d(in_ch, out_ch , kernel_size=1 , stride=1),
                SEGating(out_ch)
                ]
            )

        if batchnorm:
            self.upconv += [nn.BatchNorm3d(out_ch)]

        self.upconv = nn.Sequential(*self.upconv)

    def forward(self, x):

        return self.upconv(x)

class Conv_3d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True, batchnorm=False):

        super().__init__()
        self.conv = [nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                    SEGating(out_ch)
                    ]

        if batchnorm:
            self.conv += [nn.BatchNorm3d(out_ch)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):

        return self.conv(x)

class upConv2D(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, upmode="transpose" , batchnorm=False):

        super().__init__()

        self.upmode = upmode

        if self.upmode=="transpose":
            self.upconv = [nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)]

        else:
            self.upconv = [
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
                nn.Conv2d(in_ch, out_ch , kernel_size=1 , stride=1)
            ]

        if batchnorm:
            self.upconv += [nn.BatchNorm2d(out_ch)]

        self.upconv = nn.Sequential(*self.upconv)

    def forward(self, x):

        return self.upconv(x)


class UNet_3D_3D(nn.Module):
    def __init__(self, img_channels, block, n_inputs, n_outputs, batchnorm=False , joinType="concat" , upmode="transpose", use_uncertainty=False):
        super().__init__()

        nf = [512 , 256 , 128 , 64]        
        self.out_channels = img_channels*n_outputs
        self.joinType = joinType
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.img_channels = img_channels
        self.use_uncertainty = use_uncertainty

        growth = 2 if joinType == "concat" else 1
        self.lrelu = nn.LeakyReLU(0.2, True)

        unet_3D = importlib.import_module(".resnet_3D", "models.FLAVR")
        if n_outputs > 1:
            unet_3D.useBias = True
        self.encoder = getattr(unet_3D , block)(pretrained=False , bn=batchnorm, img_channels=img_channels)            

        self.decoder = nn.Sequential(
            Conv_3d(nf[0], nf[1] , kernel_size=3, padding=1, bias=True, batchnorm=batchnorm),
            upConv3D(nf[1]*growth, nf[2], kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1) , upmode=upmode, batchnorm=batchnorm),
            upConv3D(nf[2]*growth, nf[3], kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1) , upmode=upmode, batchnorm=batchnorm),
            Conv_3d(nf[3]*growth, nf[3] , kernel_size=3, padding=1, bias=True, batchnorm=batchnorm),
            upConv3D(nf[3]*growth , nf[3], kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1) , upmode=upmode, batchnorm=batchnorm)
        )

        self.feature_fuse = Conv_2d(nf[3]*n_inputs , nf[3]*n_inputs if use_uncertainty else nf[3], kernel_size=3 , stride=1, padding=1, batchnorm=batchnorm, bias=nn.InstanceNorm2d)
        self.feature_fuse1 = Conv_2d(nf[3]*n_inputs , nf[3]*img_channels, kernel_size=1, stride=1, batchnorm=batchnorm, bias=nn.InstanceNorm2d)
        self.tanh = torch.nn.Tanh()
        if self.use_uncertainty:
            self.uncertainty_early = Conv_2d(nf[3]*n_inputs , nf[3] , kernel_size=1 , stride=1, batchnorm=batchnorm, bias=nn.InstanceNorm2d)
            self.softmax = nn.Softmax(dim=1)
            self.uncertainty_out = nn.Conv3d(nf[3]//n_outputs, 1, kernel_size=1 , stride=1)

        self.outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(nf[3], self.out_channels , kernel_size=7 , stride=1, padding=0) 
        )

    def calc_out_patch_size(self, input_patch_size):
        x = torch.rand(tuple([1, self.img_channels] + input_patch_size)).float()
        x = x.to(next(self.parameters()).device)
        if self.use_uncertainty:
            out, _ = self(x)
        else:
            out = self(x)
        patch_size = list(out.shape[2:])
        patch_size[0] *= self.n_inputs
        return patch_size
    
    def forward(self, images, return_inetermediate_uncertainty=False, return_inetermediate_feature=False):
        # already have shape (bs, self.img_channels, d, h, w)
        # images = torch.stack(images , dim=2)

        ## Batch mean normalization works slightly better than global mean normalization, thanks to https://github.com/myungsub/CAIN
        
        # b, c, d, h, w = images.size()
        # skip_tensor = rearrange(images, 'b c d h w -> (b h) c d w')
        # skip_tensor = resize(skip_tensor, (1 / self.n_outputs, 1), order=3)
        # skip_tensor = rearrange(skip_tensor, '(b h) c d w -> b c d h w', b=b)
        
        mean_ = images[:,0:1,...].mean(2, keepdim=True).mean(3, keepdim=True).mean(4,keepdim=True)
        images[:,0:1,...] = images[:,0:1,...]-mean_

        x_0 , x_1 , x_2 , x_3 , x_4 = self.encoder(images)

        if return_inetermediate_feature:
            return x_0 , x_1 , x_2 , x_3 , x_4

        dx_3 = self.lrelu(self.decoder[0](x_4))
        dx_3 = joinTensors(dx_3 , x_3 , type=self.joinType)

        dx_2 = self.lrelu(self.decoder[1](dx_3))
        dx_2 = joinTensors(dx_2 , x_2 , type=self.joinType)

        dx_1 = self.lrelu(self.decoder[2](dx_2))
        dx_1 = joinTensors(dx_1 , x_1 , type=self.joinType)

        dx_0 = self.lrelu(self.decoder[3](dx_1))
        dx_0 = joinTensors(dx_0 , x_0 , type=self.joinType)

        dx_out = self.lrelu(self.decoder[4](dx_0))
        dx_out = torch.cat(torch.unbind(dx_out , 2) , 1)

        if self.use_uncertainty:
            dx_out = self.lrelu(self.feature_fuse(dx_out))
            out = self.feature_fuse1(dx_out)
            out = torch.split(out, dim=1, split_size_or_sections=out.shape[1]//self.n_outputs)
            out = torch.stack(out, dim=2) # bs, c(16), n_outputs, h, w

            uncertainty_early = self.uncertainty_early(dx_out)
            uncertainty_early = torch.split(uncertainty_early, dim=1, split_size_or_sections=uncertainty_early.shape[1]//self.n_outputs)
            uncertainty_early = torch.stack(uncertainty_early, dim=2)
            uncertainty_softmax = self.softmax(uncertainty_early) # bs, c(8), n_outputs, h, w
            
            out_multi = out
            out = 0
            out_img_eachs = []
            uncertainties = []
            out_seg_eachs = []
            for i in range(0, uncertainty_softmax.shape[1]):
                out_img_each = (self.tanh(out_multi[:, i*2:i*2+1, ...]) + 1) / 2
                if return_inetermediate_uncertainty:
                    out_img_eachs.append(out_img_each)
                    uncertainties.append(uncertainty_softmax[:, i:i+1])
                    out_seg_eachs.append(out_multi[:, i*2+1:i*2+2, ...])
                out_img_each = out_img_each * uncertainty_softmax[:, i:i+1]
                out_seg = out_multi[:, i*2+1:i*2+2, ...] * uncertainty_softmax[:, i:i+1]#  / uncertainty_softmax.shape[1]
                out += torch.cat([out_img_each, out_seg], dim=1)
        else:
            out = self.lrelu(self.feature_fuse(dx_out))
            out = self.outconv(out)
            out = torch.split(out, dim=1, split_size_or_sections=self.img_channels)
        
            mean_ = mean_.squeeze(2)
            if self.img_channels > 1:
                out = [torch.cat([torch.tanh(o[:, 0:1,...]+mean_), o[:, 1:2,...]], dim=1) for o in out]
            else:
                out = [o+mean_ for o in out]
            out = torch.stack(out, dim=2)
        # out = out_residual + skip_tensor[:,:,int(self.n_outputs)*(self.n_inputs//2-1):int(self.n_outputs)*(self.n_inputs//2),...]
 
        if return_inetermediate_uncertainty:
            return out_img_eachs, uncertainties, out_seg_eachs

        if self.use_uncertainty:
            uncertainty_out = torch.nn.Sigmoid()(self.uncertainty_out(uncertainty_softmax))
            return out, uncertainty_out
        else:
            return out