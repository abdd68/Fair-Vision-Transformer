# This code is modified from https://github.com/huggingface/transformers/tree/master/examples/research_projects/movement-pruning
# Licensed under the Apache License, Version 2.0 (the "License");
# We add more functionalities as well as remove unnecessary functionalities
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from .binarizer import TopKBinarizer

class MaskedLinear(nn.Linear):
    """
    Fully Connected layer with on the fly adaptive mask during training.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mask_init: str = "constant",
        mask_scale: float = 0.0,
        head_split: int = -1,
        bias_mask: bool = False,
        head_masking: bool = False,
        fc_masking: bool = False,
        threshold_init = 10.0,
        num_classes = None,
        mask_numbers = 2
    ):

        super(
            MaskedLinear,
            self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias)
        self.threshold_init = threshold_init
        self.head_split = head_split
        self.bias_mask = bias_mask
        self.head_masking = head_masking
        self.fc_masking = fc_masking

        self.inference_mode = False

        self.mask_scale = mask_scale
        self.mask_init = mask_init
        self.head_saliency_scores = None
        self.saliency_scores = None
        self.num_classes = num_classes
        self.mask_numbers = mask_numbers
        if self.fc_masking:
            self.saliency_scores = nn.Parameter(
                torch.Tensor(
                    self.weight.size(0), self.mask_numbers))  # number of output * 1 # torch.Size([768, 1])
            self.init_mask(self.saliency_scores) # 将mask全部置零
            self.threshold_fc = nn.Parameter(torch.zeros(1) + threshold_init) # threshold_init: 10
        if self.head_masking:
            self.head_saliency_scores = nn.Parameter(
                torch.Tensor(self.head_split, self.mask_numbers))  # number of heads * 1 # torch.Size([12, 1])
            self.init_mask(self.head_saliency_scores)
            self.threshold_head = nn.Parameter((torch.zeros(1) + threshold_init).cuda()) # threshold_init: 10

    def init_mask(self, mask):
        if self.mask_init == "constant":
            init.constant_(mask, val=self.mask_scale) # self.mask_scale: 0
        elif self.mask_init == "uniform":
            init.uniform_(mask, a=-self.mask_scale, b=self.mask_scale)
        elif self.mask_init == "kaiming":
            init.kaiming_uniform_(mask, a=math.sqrt(5))

    def get_mask(self, args, sa, ma):
        # get head mask
        if self.head_masking:
            mask_head, split_line = TopKBinarizer.apply(
                args, self.head_saliency_scores, self.threshold_head, -1, sa, ma)  # for now, only support this # [12,1]
        else:
            mask_head = None
        if self.fc_masking:
            mask, split_line = TopKBinarizer.apply(
                args, self.saliency_scores, self.threshold_fc, -1, sa, ma) # [768,1]
            
        else:
            mask = None


        return mask_head, mask, split_line

    def forward(self, args, input, sa, ma):
        if (not self.inference_mode and args != None):
            output = self.training_forward(args, input, sa, ma)
        else:
            output = self.inference_forward(input)
        return output


    def inference_forward(self, input: torch.tensor):
        return F.linear(input, self.weight, self.bias)

    def training_forward(self, args, input: torch.tensor, sa, ma): # input: [8, 197, 768]
        if(sa is None):
            if(self.weight.shape[0] == 3 * self.weight.shape[1]):
                mask_head, mask = torch.ones((args.num_heads,1)).cuda(), None
            else:
                mask_head, mask =  None, torch.ones((self.weight.shape[0],1)).cuda()
        else:
            mask_head, mask, _ = self.get_mask(args, sa, ma)
        weight_shape = self.weight.size()
        bias_shape = self.bias.size()
        if self.head_masking: 
            weight_thresholded = self.weight * mask_head.repeat_interleave(64).unsqueeze(1).repeat(3, 1)
            
            if self.bias_mask: 
                bias_thresholded = self.bias * mask_head.repeat_interleave(64).unsqueeze(1).repeat(3, 1).view(bias_shape)
        else: 
            weight_thresholded = self.weight 
            bias_thresholded = self.bias
        # Mask weights with computed mask
        if self.fc_masking:
            weight_thresholded = mask * weight_thresholded # mask: [768, 1]; weight_thresholded: [2304, 768]; bias_threshold: [2304]
            if self.bias_mask:
                bias_thresholded = mask.view(
                    self.bias.size()) * bias_thresholded
            else:
                bias_thresholded = bias_thresholded
        return F.linear(input, weight_thresholded, bias_thresholded) # [8, 197, 2304]
