import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

class FlowNetEncoderRefineMultiscale(nn.Module):
    def __init__(self, args, input_channels = 12, batchNorm=True, div_flow=20):
        super(FlowNetEncoderRefineMultiscale, self).__init__()
        
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
        self.deconv4 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.conv_pred4 = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.flow_upsample4 = nn.Upsample(scale_factor=2)
        self.deconv3 = nn.ConvTranspose2d(514, 256, kernel_size=4, stride=2, padding=1)
        self.conv_pred3 = nn.Conv2d(514, 2, kernel_size=3, stride=1, padding=1)
        self.flow_upsample3 = nn.Upsample(scale_factor=2)
        self.conv_pred2 = nn.Sequential(
            nn.Conv2d(386, 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=4)
        )

        ''''''

    def forward(self, inputs):
        ## input normalization
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat( (x[:,:,0,:,:], x[:,:,1,:,:]), dim = 1)
        x = self.conv1(x)
        down2 = self.conv2(x)
        down3 = self.conv3(down2)
        down4 = self.conv4(down3)
        flow4 = self.conv_pred4(down4)
        up3 = torch.cat((down3, self.deconv4(down4), self.flow_upsample4(flow4)), dim=1)
        flow3 = self.conv_pred3(up3)
        up2 = torch.cat((down2, self.deconv3(up3), self.flow_upsample3(flow3)), dim=1)
        flow2 = self.conv_pred2(up2)

        ##

        '''Implement Codes here'''
        ''''''

        if self.training:
            return flow2, flow3, flow4
        else:
            return flow2 * self.div_flow
