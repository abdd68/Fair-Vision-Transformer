"""
Implements the loss function
"""
import torch
import logging
import math
logger = logging.getLogger(__name__)

class LossFunction(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra distance loss to maintain accuracy.
    """
    def __init__(self, base_criterion: torch.nn.Module, model: torch.nn.Module, alpha):
        super().__init__()
        self.base_criterion = base_criterion
        self.model = model
        self.alpha = alpha

    def forward(self, args, outputs, labels, weight_bias):
        """
        Args:
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        if not isinstance(outputs, torch.Tensor):
            outputs, outputs_classifier = outputs
        base_loss = self.base_criterion(outputs, labels)
        for i in range(len(outputs_classifier)):
            base_loss += self.base_criterion(outputs_classifier[i], labels)
        dist_loss = self.alpha * distance_loss(args, outputs, labels, weight_bias)
        loss = base_loss + dist_loss
        # logger.info(f"baseloss:{base_loss.item():.4f}; distloss:{dist_loss:.5f}")
        return loss


def remain_param_compute(threshold_list):
    output = 0.

    attn, o_matrix, fc1, fc2 = threshold_list

    output += torch.max(attn, torch.tensor(1 / 12.)).type(fc1.type()) * 3
    output += torch.max(attn, torch.tensor(1 / 12.)).type(fc1.type()) * torch.max(o_matrix, torch.tensor(1 / 768.)).type(fc1.type())
    output += torch.max(fc1, torch.tensor(1 / 3072.)).type(fc1.type()) * 4
    output += torch.max(fc1, torch.tensor(1 / 3072.)).type(fc1.type()) * torch.max(fc2, torch.tensor(1 / 768.)).type(fc1.type()) * 4
    
    return output

def distance_loss(args, outputs, labels, weight_bias):
    loss_cal = 0
    if(weight_bias == None):
        return 0
    (theta_1, theta_2, b) = weight_bias
    x, y  = torch.Tensor(outputs.shape[0]), torch.Tensor(outputs.shape[0])
    for i in range(outputs.shape[0]):
        x[i] = outputs[i,labels[i]]
        tmp = outputs[i].topk(2)[0]

        for j in range(tmp.shape[0]):
            if(tmp[j] == x[i]):
                tmp = torch.cat((tmp[:j], tmp[j+1:]))
                break
        y[i] = tmp.sum()

        # regularizer formula: y = (-theta_1 / theta_2) * x - b / theta_2
        dist = (theta_1 * x[i] + theta_2 * y[i] + b - args.bias) / math.sqrt(theta_1**2 + theta_2**2)
        if(dist >= 10):
            loss_cal += -2
        elif(dist >= 0):
            loss_cal += -args.gamma * dist
        else:
            loss_cal += - dist
    
    return loss_cal
