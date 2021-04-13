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
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=4)
        )
        ''''''

    def forward(self, inputs):
        ## input normalization
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat( (x[:,:,0,:,:], x[:,:,1,:,:]), dim = 1)
        ##
        '''Implement Codes here'''
        x = self.conv1(x)
        flow2 = self.conv2(x)
        flow3 = self.conv3(flow2)
        flow4 = self.conv4(flow3)
        flow3 = torch.cat((flow3, self.deconv1(flow4)), dim=1)
        flow2 = torch.cat((flow2, self.deconv2(flow3)), dim=1)
        flow2 = self.conv5(flow2)

        ''''''

        if self.training:
            return flow2
        else:
            return flow2 * self.div_flow
