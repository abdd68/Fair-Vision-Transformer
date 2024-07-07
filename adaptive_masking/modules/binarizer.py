import torch
from torch import autograd
import math
from torch import nn
import logging
logger = logging.getLogger(__name__)
class TopKBinarizer(autograd.Function):
    @staticmethod
    def forward(ctx, args, saliency_scores: torch.tensor, threshold: torch.tensor, head_split: int, sa, ma): # saliency_scores:(output/head,1), threshold: scalar
        
        # Get the subnetwork by sorting the saliency_scores and using the top threshold
        threshold = torch.sigmoid(threshold).item()

        saliency_scores_sum = torch.ones_like(saliency_scores[:,:1])
        for j in range(len(args.varsigma)):
            saliency_scores_sum += args.varsigma[j] * saliency_scores[:,j:j+1]

        mask_sum = saliency_scores_sum.clone() # mask:(output, 1)

        j = math.ceil(threshold * saliency_scores_sum.numel())
        v = 0

        base_number = 16 # constant
        if j % base_number != 0 and saliency_scores_sum.size()[0] % base_number == 0: # j is the saperator
            if j > base_number:
                v = j - j % base_number
                j = v
                if j > saliency_scores_sum.size()[0]: # robustness
                    j = saliency_scores_sum.size()[0]
            else:
                j = base_number
        
        ctx.save_for_backward(mask_sum, saliency_scores)
        ctx.args = args
        ctx.num_classes, ctx.sa, ctx.ma  = saliency_scores.shape[1], sa, ma
        saliency_scores = torch.clamp(saliency_scores,min = -1., max = 1.)
        return mask_sum, j

    @staticmethod
    def backward(ctx, gradOutput, gradOutput2):
        mask_sum, masks,  = ctx.saved_tensors # [768, num_classes], [batch_size]
        args = ctx.args
        num_classes, sa, ma = ctx.num_classes, ctx.sa, ctx.ma

        gradOutput_tmp = torch.zeros((mask_sum.shape[0],num_classes)).cuda()
        if(not args.manual):
            for i in range(ma.shape[0]):
                j = ma[i]
                gradOutput_tmp[:,j:j+1] += args.varsigma[j] * gradOutput / sa.shape[0]
            for i in range(ma.shape[0]):
                j = ma[i]
                args.v_grad[j] += (gradOutput[:,0] * (masks[:,j])).sum() / sa.shape[0]
        
        return None, gradOutput_tmp, ((gradOutput * mask_sum).sum()).view(-1), None, None, None