import torch
import torch.nn as nn
import numpy as np  

from models.backbones.resnet import ResNet_C5_Dilation

__all__ = ["RPRCNN"]

class RPRCNN(nn.Module):
    def __init__(self, is_train=True):
        super().__init__()

        # Backbone
        self.Conv_Body = ResNet_C5_Dilation()
        self.dim_in = self.Conv_Body.dim_out
        self.spatial_scale = self.Conv_Body.spatial_scale

        # FPN
        

        # SemSeg

        # RPN

        # FastRCNN

        # Parsing

        # ReScore
        
    def forward(self, images, targets):
        pass