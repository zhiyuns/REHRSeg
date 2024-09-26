import math
import numpy as np
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder

from einops import rearrange, repeat

class MyUnetDecoder(UNetDecoder):
    def __init__(self,
                 encoder,
                 num_classes,
                 n_conv_per_stage,
                 deep_supervision,
                 deep_features,
                 nonlin_first = False
                 ):
        super(MyUnetDecoder, self).__init__(encoder, num_classes, n_conv_per_stage, deep_supervision, nonlin_first)
        self.deep_features = deep_features

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        feature_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            if self.deep_features and s == (len(self.stages) - 1):
                feature_outputs = x
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        
        if self.deep_features:
            return r, feature_outputs
        else:
            return r

def cosine_distance_loss(tensor1, tensor2):
    # Normalize tensors
    tensor1 = torch.nn.functional.normalize(tensor1, p=2, dim=1)
    tensor2 = torch.nn.functional.normalize(tensor2, p=2, dim=1)

    # Reshape tensors
    tensor1 = tensor1.reshape(tensor1.shape[0], tensor1.shape[1], -1)
    tensor2 = tensor2.reshape(tensor2.shape[0], tensor2.shape[1], -1)

    # Calculate cosine similarity
    cosine_similarity = torch.cosine_similarity(tensor1, tensor2, dim=2)

    # Calculate cosine distance
    cosine_distance = 1 - cosine_similarity

    # Calculate average cosine distance
    average_cosine_distance = cosine_distance.mean()

    return average_cosine_distance

def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8

def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
    return torch.einsum('icm,icn->imn', [feat, feat])

def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis

class CriterionPairWiseforWholeFeatAfterPool(nn.Module):
    def __init__(self, scale):
        '''inter pair-wise loss from inter feature maps'''
        super(CriterionPairWiseforWholeFeatAfterPool, self).__init__()
        self.criterion = sim_dis_compute
        self.scale = scale

    def forward(self, preds_S, preds_T):
        feat_S = preds_S
        feat_T = preds_T
        feat_T.detach()
        _,_,s,total_w,total_h = feat_S.shape
        feat_S = rearrange(feat_S, 'b c s h w -> (b s) c h w')
        feat_T = rearrange(feat_T, 'b c s h w -> (b s) c h w')
        
        patch_w, patch_h = int(total_w*self.scale), int(total_h*self.scale)
        maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True) # change
        loss = self.criterion(maxpool(feat_S), maxpool(feat_T)) / s
        return loss

class Distiller(nn.Module):
    def __init__(self, student_dim, teacher_dim, lambda_l1=0.0, lambda_cosine=0.0, lambda_structure=0.0):
        super(Distiller, self).__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_cosine = lambda_cosine
        self.lambda_structure = lambda_structure
        if lambda_structure > 0.0:
            pool_scale = 0.5
            self.criterion_structure = CriterionPairWiseforWholeFeatAfterPool(scale=pool_scale)
        # self.distill = nn.Identity()
        # self.distill = nn.Sequential(*[nn.Conv3d(in_channels=student_dim, out_channels=student_dim, kernel_size=1, stride=1, padding=0),
        #                               nn.InstanceNorm3d(student_dim),
        #                               nn.ReLU(),
        #                               nn.Conv3d(in_channels=student_dim, out_channels=teacher_dim, kernel_size=1, stride=1, padding=0),])
        # self.distill = nn.Sequential(*[nn.Linear(student_dim, student_dim),
        #                               nn.LayerNorm(student_dim),
        #                               nn.GELU(),
        #                               nn.Linear(student_dim, teacher_dim),])
        self.distill = nn.Conv3d(in_channels=student_dim, out_channels=teacher_dim, kernel_size=1, stride=1, padding=0)
    def forward(self, feature_student, feature_teacher):
        loss = 0
        if self.lambda_structure > 0:
            loss_structure = self.criterion_structure(feature_student, feature_teacher)
            loss += self.lambda_structure * loss_structure
        # feature_student = rearrange(feature_student, 'b c s h w -> b (s h w) c')
        # feature_teacher = rearrange(feature_teacher, 'b c s h w -> b (s h w) c')
        distilled_feature = self.distill(feature_student)
        # smooth l1
        if self.lambda_l1 > 0:
            loss += F.smooth_l1_loss(distilled_feature, feature_teacher) * self.lambda_l1
        if self.lambda_cosine > 0:
            # distilled_feature = rearrange(distilled_feature, 'b d c -> b c d')
            # feature_teacher = rearrange(feature_teacher, 'b d c -> b c d')
            loss_cosine = cosine_distance_loss(distilled_feature, feature_teacher)
            loss += self.lambda_cosine * loss_cosine
        
        return loss

class SegModel(PlainConvUNet):
    def __init__(self, 
                 input_channels,
                 n_stages,
                 features_per_stage,
                 conv_op,
                 kernel_sizes,
                 strides,
                 n_conv_per_stage,
                 num_classes,
                 upscale,
                 n_conv_per_stage_decoder,
                 conv_bias = False,
                 norm_op = None,
                 norm_op_kwargs = None,
                 dropout_op = None,
                 dropout_op_kwargs = None,
                 nonlin = None,
                 nonlin_kwargs = None,
                 deep_supervision = False,
                 nonlin_first = False):
        super(SegModel, self).__init__(input_channels, 
                                       n_stages,
                                       features_per_stage,
                                       conv_op,
                                       kernel_sizes,
                                       strides,
                                       n_conv_per_stage,
                                       num_classes,
                                       n_conv_per_stage_decoder,
                                       conv_bias,
                                       norm_op,
                                       norm_op_kwargs,
                                       dropout_op,
                                       dropout_op_kwargs,
                                       nonlin,
                                       nonlin_kwargs,
                                       deep_supervision,
                                       nonlin_first)
        
        self.decoder = MyUnetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision, deep_features=True,
                                   nonlin_first=nonlin_first)
        
        self.upscale = upscale
        self.sr_head = nn.Sequential(*[nn.Conv3d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.Conv3d(in_channels=16, out_channels=num_classes, kernel_size=5, stride=1, padding=2)])

    def forward(self, x, return_inetermediate_feature=False):
        skips = self.encoder(x)
        out, features = self.decoder(skips)
        out_up = F.interpolate(features, scale_factor=(self.upscale, 1, 1), mode='trilinear', align_corners=True)
        out_up = self.sr_head(out_up)

        if return_inetermediate_feature:
            return out, out_up, skips
        else:
            return out, out_up
        