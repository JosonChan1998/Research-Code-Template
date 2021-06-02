import torch
import torch.nn as nn

import models.ops as ops

def make_norm(c, norm='bn', eps=1e-5, an_k=10):
    if norm == 'bn':
        return nn.BatchNorm2d(c, eps=eps)
    elif norm == 'gn':
        group = 32 if c >= 32 else c
        assert c % group == 0
        return nn.GroupNorm(group, c, eps=eps)
    elif norm == 'none':
        return None
    else:
        return nn.BatchNorm2d(c, eps=eps)

def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False

def make_conv(in_channels, out_channels, kernel=3, stride=1, dilation=1, padding=None, groups=1, 
              use_dwconv=False, conv_type='normal', use_bn=False, use_gn=False, use_relu=False, 
              kaiming_init=True, suffix_1x1=False, inplace=True, eps=1e-5, gn_group=32):
    _padding = (dilation * kernel - dilation) // 2 if padding is None else padding
    if use_dwconv:
        assert in_channels == out_channels
        _groups = out_channels
    else:
        _groups = groups
    
    if conv_type == 'normal':
        conv_op = nn.Conv2d
    elif conv_type == 'deform':
        conv_op = ops.DeformConvPack
    elif conv_type == 'deformv2':
        conv_op = ops.ModulatedDeformConvPack
    elif conv_type == 'convws':
        conv_op = ops.Conv2dWS
    else:
        raise ValueError('{} type conv operation is not supported.'.format(conv))
    conv = conv_op(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=_padding,
                   dilation=dilation, groups=_groups, bias=False if use_bn or use_gn else True)
    if kaiming_init:
        nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
    else:
        torch.nn.init.normal_(conv.weight, std=0.01)
    if not (use_bn or use_gn):
        nn.init.constant_(conv.bias, 0)
    module = [conv, ]
    
    if use_bn:
        module.append(nn.BatchNorm2d(out_channels, eps=eps))
    if use_gn:
        module.append(nn.GroupNorm(gn_group, out_channels, eps=eps))
    if use_relu:
        module.append(nn.ReLU(inplace=inplace))
        
    if suffix_1x1:
        module.append(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False if use_bn or use_gn else True)
        )
        if use_bn:
            module.append(nn.BatchNorm2d(out_channels, eps=eps))
        if use_gn:
            module.append(nn.GroupNorm(gn_group, out_channels, eps=eps))
        if use_relu:
            module.append(nn.ReLU(inplace=inplace))
    if len(module) > 1:
        return nn.Sequential(*module)
    return conv