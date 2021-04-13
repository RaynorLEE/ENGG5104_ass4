import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EPELoss(nn.Module):
    def __init__(self, args, div_flow = 0.05):
        super(EPELoss, self).__init__()
        self.div_flow = div_flow 
        self.loss_labels = ['EPE'],

    def forward(self, output, target):
        epevalue = 0
        target = self.div_flow * target
        assert output.shape == target.shape, (output.shape, target.shape)
        ''' Implement the EPE loss here'''
        # B, C, H, W = output.shape
        # diff = output - target
        # square_diff = diff * diff
        # for i in range(B):
        #     curr_epevalue = 0
        #     for u in range(H):
        #         for v in range(W):
        #             curr_epevalue += math.sqrt(square_diff[i][0][u][v] + square_diff[i][1][u][v])
        #     curr_epevalue /= H * W
        #     epevalue += curr_epevalue
        #
        # print(epevalue)
        # ref = torch.norm(target - output, p=2, dim=1)
        # print(ref)
        # ''''''
        # return [epevalue]
        epevalue = torch.norm(target - output, p=2, dim=1).mean()
        return [epevalue]


class MultiscaleLoss(nn.Module):
    def __init__(self, args):
        super(MultiscaleLoss, self).__init__()

        self.args = args
        self.div_flow = 0.05
        self.loss_labels = ['Multiscale'],
        ''' Implement the MultiScale loss here'''
        #   Reference: https://arxiv.org/pdf/1709.02371.pdf
        #   self.loss_weights = torch.FloatTensor([0.32, 0.04, 0.01], device=)
        self.loss_weights = tuple([0.32, 0.16, 0.08])
        ''''''

    def forward(self, output, target):
        lossvalue = 0
        epevalue = 0
        target = self.div_flow * target
        for i, output_ in enumerate(output):
            target_ = F.interpolate(target, output_.shape[2:], mode='bilinear', align_corners=False)
            assert output_.shape == target_.shape, (output_.shape, target_.shape)
            ''' Implement the MultiScale loss here'''
            epevalue += self.loss_weights[i] * torch.norm(target_ - output_, p=2, dim=1).mean()
            ''''''
        return [epevalue]
