import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

class FlowNetEncoderRefine(nn.Module):
    def __init__(self, args, input_channels = 12, batchNorm=True, div_flow=20):
        super(FlowNetEncoderRefine, self).__init__()
        
        self.rgb_max = args.rgb_max
        self.div_flow = div_flow    # A coefficient to obtain small output value for easy training, ignore it

        '''Implement Codes here'''
        ''''''

    def forward(self, inputs):
        ## input normalization
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat( (x[:,:,0,:,:], x[:,:,1,:,:]), dim = 1)
        ##
        '''Implement Codes here'''
        ''''''

        if self.training:
            return flow2
        else:
            return flow2 * self.div_flow
