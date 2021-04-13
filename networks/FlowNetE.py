import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np


class FlowNetEncoder(nn.Module):
    def __init__(self, args, input_channels = 6, div_flow=20):
        super(FlowNetEncoder,self).__init__()

        self.rgb_max = args.rgb_max
        self.div_flow = div_flow      # A coefficient to obtain small output value for easy training, ignore it
        '''Implement Codes here'''
        ''''''
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
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=16)
        )
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
        #     nn.LeakyReLU()
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        #     nn.LeakyReLU()
        # )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        #     nn.LeakyReLU()
        # )
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
        #     nn.LeakyReLU()
        # )
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1),
        #     nn.Upsample(scale_factor=16)
        # )

    def forward(self, inputs):
        ## input normalization
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat( (x[:,:,0,:,:], x[:,:,1,:,:]), dim = 1)
        ##
        '''Implement Codes here'''
        ''''''
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        flow4 = self.conv5(x)

        if self.training:
            return flow4
        else:
            return flow4 * self.div_flow

