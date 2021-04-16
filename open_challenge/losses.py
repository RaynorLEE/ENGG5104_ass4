import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class OursLoss(nn.Module):
    def __init__(self, args, div_flow = 0.05):
        super(OursLoss, self).__init__()
        self.div_flow = div_flow 
        self.loss_labels = ['Ours'],
        ''' Implement the MultiScale loss here'''
        #   Reference: https://arxiv.org/pdf/1709.02371.pdf
        self.loss_weights = tuple([0.32, 0.16, 0.08, 0.04])

    def forward(self, output, target):
        # epevalue = 0
        # target = self.div_flow * target
        # assert output.shape == target.shape, (output.shape, target.shape)
        # ''' Implement the your loss here'''
        # ''''''
        # return [epevalue]
        epevalue = 0
        target = self.div_flow * target
        for i, output_ in enumerate(output):
            target_ = F.interpolate(target, output_.shape[2:], mode='bilinear', align_corners=False)
            assert output_.shape == target_.shape, (output_.shape, target_.shape)
            ''' Implement the MultiScale loss here'''
            epevalue += self.loss_weights[i] * torch.norm(target_ - output_, p=2, dim=1).mean()
            ''''''
        return [epevalue]
