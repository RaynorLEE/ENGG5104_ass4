import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np



class FlowNetOurs(nn.Module):
    def __init__(self, args, input_channels=12, batchNorm=True, div_flow=20):
        super(FlowNetOurs, self).__init__()

        self.rgb_max = args.rgb_max
        self.div_flow = div_flow  # A coefficient to obtain small output value for easy training, ignore it

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
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
        )
        self.deconv5 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.conv_pred5 = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.flow_upsample5 = nn.Upsample(scale_factor=2)
        self.deconv4 = nn.ConvTranspose2d(514, 256, kernel_size=4, stride=2, padding=1)
        self.conv_pred4 = nn.Conv2d(514, 2, kernel_size=3, stride=1, padding=1)
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
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat((x[:, :, 0, :, :], x[:, :, 1, :, :]), dim=1)
        x = self.conv1(x)
        down2 = self.conv2(x)
        down3 = self.conv3(down2)
        down4 = self.conv4(down3)
        down5 = self.conv5(down4)
        flow5 = self.conv_pred5(down5)
        up4 = torch.cat((down4, self.deconv5(down5), self.flow_upsample5(flow5)), dim=1)
        flow4 = self.conv_pred4(up4)
        up3 = torch.cat((down3, self.deconv4(up4), self.flow_upsample4(flow4)), dim=1)
        flow3 = self.conv_pred3(up3)
        up2 = torch.cat((down2, self.deconv3(up3), self.flow_upsample3(flow3)), dim=1)
        flow2 = self.conv_pred2(up2)

        ##

        '''Implement Codes here'''
        ''''''

        if self.training:
            return flow2, flow3, flow4, flow5
        else:
            return flow2 * self.div_flow


# class FlowNetOurs(nn.Module):
#     def __init__(self, args, input_channels = 6, div_flow=20):
#         super(FlowNetOurs, self).__init__()
#
#         self.rgb_max = args.rgb_max
#         self.div_flow = div_flow    # A coefficient to obtain small output value for easy training, ignore it
#
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
#             nn.LeakyReLU()
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
#             nn.LeakyReLU()
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
#             nn.LeakyReLU()
#         )
#         self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=15, stride=1)
#         self.corr_act = nn.LeakyReLU()
#         self.conv_redir = nn.Sequential(
#             nn.Conv2d(128, 32, kernel_size=1, stride=1),
#             nn.LeakyReLU()
#         )
#         self.conv3_1 = nn.Sequential(
#             nn.Conv2d(257, 256, kernel_size=3, stride=1, padding=1)
#         )
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.LeakyReLU(),
#             nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
#             nn.LeakyReLU()
#         )
#         self.deconv4 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
#         self.conv_pred4 = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
#         self.flow_upsample4 = nn.Upsample(scale_factor=2)
#         self.deconv3 = nn.ConvTranspose2d(514, 256, kernel_size=4, stride=2, padding=1)
#         self.conv_pred3 = nn.Conv2d(514, 2, kernel_size=3, stride=1, padding=1)
#         self.flow_upsample3 = nn.Upsample(scale_factor=2)
#         self.conv_pred2 = nn.Sequential(
#             nn.Conv2d(322, 2, kernel_size=3, stride=1, padding=1),
#             nn.Upsample(scale_factor=4)
#         )
#
#     def forward(self, inputs):
#         ## input normalization
#         rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
#         x = (inputs - rgb_mean) / self.rgb_max
#         x = torch.cat( (x[:,:,0,:,:], x[:,:,1,:,:]), dim = 1)
#
#         x1 = x[:, 0:3, :, :]
#         x2 = x[:, 3:, :, :]
#         down1a = self.conv1(x1)
#         down2a = self.conv2(down1a)
#         down3a = self.conv3(down2a)
#
#         down1b = self.conv1(x2)
#         down2b = self.conv2(down1b)
#         down3b = self.conv3(down2b)
#
#         corr = self.corr(down3a, down3b)
#         corr = torch.flatten(corr, start_dim=1, end_dim=2)
#         corr = self.corr_act(corr)
#
#         down_redir = self.conv_redir(down3a)
#         down3_1 = self.conv3_1(torch.cat((down_redir, corr), 1))
#         down4 = self.conv4(down3_1)
#
#         flow4 = self.conv_pred4(down4)
#         up3 = torch.cat((down3_1, self.deconv4(down4), self.flow_upsample4(flow4)), dim=1)
#         flow3 = self.conv_pred3(up3)
#         up2 = torch.cat((down2a, self.deconv3(up3), self.flow_upsample3(flow3)), dim=1)
#         flow2 = self.conv_pred2(up2)
#
#         if self.training:
#             return flow2, flow3, flow4
#         else:
#             return flow2 * self.div_flow

