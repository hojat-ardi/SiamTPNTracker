"""
Basic STARK Model (Spatial-only).
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import sparse_
from typing import Dict, List

from .backbone import build_backbone
from .head import build_head
from .fpn import build_fpn
#torch.autograd.set_detect_anomaly(True)

class SiamTPN(nn.Module):
    """ This is the base class for Transformer Tracking """
    def __init__(self, backbone, fpn, head, cfg):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries.
        """
        super().__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.head = head
        self.cfg = cfg


    def forward(self, train_imgs, test_img):
        """
        templates: num_images, B, C, H, W
        search_img: B, C, H, W
        """
        train_feat = self.backbone(train_imgs)
        test_feat = self.backbone(test_img)
        train_feat = self.fpn(train_feat)
        test_feat = self.fpn(test_feat)
        
        return  self.head(test_feat, train_feat)

# ------------------LINK TO /home/ardi/Desktop/project/SiamTPNTracker/model_detail.py ----------------------
    # def forward(self, train_imgs, test_img):
    #     print("Input Train Images Shape:", train_imgs.shape)
    #     print("Input Test Image Shape:", test_img.shape)

    #     train_feat = self.backbone(train_imgs)
    #     if isinstance(train_feat, tuple):  # Checking if output is a tuple
    #         train_feat = train_feat[0]  # Assuming we need the first element
    #     print("Output Backbone Train Shape:", train_feat.shape)

    #     test_feat = self.backbone(test_img)
    #     if isinstance(test_feat, tuple):  # Checking if output is a tuple
    #         test_feat = test_feat[0]  # Assuming we need the first element
    #     print("Output Backbone Test Shape:", test_feat.shape)

    #     train_feat = self.fpn(train_feat)
    #     if isinstance(train_feat, tuple):
    #         train_feat = train_feat[0]
    #     print("Output FPN Train Shape:", train_feat.shape)

    #     test_feat = self.fpn(test_feat)
    #     if isinstance(test_feat, tuple):
    #         test_feat = test_feat[0]
    #     print("Output FPN Test Shape:", test_feat.shape)

    #     final_output = self.head(test_feat, train_feat)
    #     if isinstance(final_output, tuple):
    #         final_output = final_output[0]
    #     print("Output Head Shape:", final_output.shape)

    #     return final_output
    # ----------------------------------------------------------------------------------------

def build_network(cfg):
    backbone = build_backbone(cfg)
    fpn = build_fpn(cfg)
    head = build_head(cfg)
    
    model = SiamTPN(
        backbone,
        fpn,
        head=head,
        cfg = cfg
    )
    return model
