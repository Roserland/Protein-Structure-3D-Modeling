from __future__ import print_function
from tkinter.messagebox import NO
from turtle import forward
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

class Residual_3D(nn.Module):
    def __init__(self, inp, oup, stride, expanse_ratio, relu=None):
        """
            inp: in channels;
            out: output channels
            stride: eg: tuple(1, 1, 1)
            expanse_ratio: whether add an additional Conv3D
        """
        super(Residual_3D, self).__init__()
        self.stride = stride
        hidden_dim = round(inp * expanse_ratio)
        self.use_res_connect = self.stride == (1,1,1) and inp == oup

        if relu is None:
            self.relu = nn.ReLU6(inplace=True)
        else:
            self.relu = relu

        if expanse_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                self.relu,
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                self.relu,
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                self.relu,
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class HourGlass3D(nn.Module):
    def __init__(self, in_channels) -> None:
        super(HourGlass3D, self).__init__()
        self.expanse_ratio = 2
        
        # down-sample layers
        self.conv1 = Residual_3D(in_channels, in_channels * 2, 2, self.expanse_ratio)

        self.conv2 = Residual_3D(in_channels * 2, in_channels * 2, 1, self.expanse_ratio)

        self.conv3 = Residual_3D(in_channels * 2, in_channels * 4, 2, self.expanse_ratio)

        self.conv4 = Residual_3D(in_channels * 4, in_channels * 4, 1, self.expanse_ratio)

        # up-sample layers
        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = Residual_3D(in_channels, in_channels, 1, self.expanse_ratio)
        self.redir2 = Residual_3D(in_channels * 2, in_channels * 2, 1, self.expanse_ratio)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6
        

class HourGlass3DNet_2(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, regression=False) -> None:
        super(HourGlass3DNet_2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = 4 # out_channels
        self.regression = regression

        self.relu_0 = nn.LeakyReLU()
        self.conv_0 = Residual_3D(in_channels, out_channels, (1, 1, 1), 1, relu=self.relu_0)
    
        self.hourglass_1 = HourGlass3D(in_channels=self.out_channels)
        self.hourglass_2 = HourGlass3D(in_channels=self.out_channels)

        # if regerssion
        self.ave_pooling = nn.AvgPool3d(2, 2, )
        self.fcn_Ca_pos = self.fcn(8**3, 3)
        self.fcn_N_pos = self.fcn(8**3, 3)
        self.fcn_C_pos = self.fcn(8**3, 3)
        self.fcn_O_pos = self.fcn(8**3, 3)

        
    def fcn(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(feat_in, feat_out)
        )

    def forward(self, x):
        out_0 = self.conv_0(x)
        out_1 = self.hourglass_1(out_0)
        out_2 = self.hourglass_2(out_1)
        if self.regression:
            # fea = self.ave_pooling(out_2).reshape(out_2.shape[0], -1)
            ca_fea = self.ave_pooling(out_2[:, 0])
            n_fea = self.ave_pooling(out_2[:, 1])
            c_fea = self.ave_pooling(out_2[:, 2])
            o_fea = self.ave_pooling(out_2[:, 3])
            Ca_output = self.fcn_Ca_pos(ca_fea.reshape(ca_fea.shape[0], -1))
            N_output = self.fcn_N_pos(n_fea.reshape(n_fea.shape[0], -1))
            C_output = self.fcn_C_pos(c_fea.reshape(c_fea.shape[0], -1))
            O_output = self.fcn_O_pos(o_fea.reshape(o_fea.shape[0], -1))
            return Ca_output, N_output, C_output, O_output
        else:
            return out_2



class HourGlass3DNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, regression=False) -> None:
        super(HourGlass3DNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.regression = regression

        self.relu_0 = nn.LeakyReLU()
        self.conv_0 = Residual_3D(in_channels, out_channels, (1, 1, 1), 1, relu=self.relu_0)
    
        self.hourglass_1 = HourGlass3D(in_channels=self.out_channels)
        self.hourglass_2 = HourGlass3D(in_channels=self.out_channels)
        self.hourglass_3 = HourGlass3D(in_channels=self.out_channels)

        # if regerssion
        self.ave_pooling = nn.AvgPool3d(2, 2, )
        self.fcn_Ca_pos = self.fcn(8**3, 3)
        self.fcn_N_pos = self.fcn(8**3, 3)
        self.fcn_C_pos = self.fcn(8**3, 3)
        self.fcn_O_pos = self.fcn(8**3, 3)

        

    def fcn(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(feat_in, feat_out)
        )

    def forward(self, x):
        out_0 = self.conv_0(x)
        out_1 = self.hourglass_1(out_0)
        out_2 = self.hourglass_2(out_1)
        out_2 = self.hourglass_3(out_2)
        if self.regression:
            fea = self.ave_pooling(out_2).reshape(out_2.shape[0], -1)
            Ca_output = self.fcn_Ca_pos(fea)
            N_output = self.fcn_N_pos(fea)
            C_output = self.fcn_C_pos(fea)
            O_output = self.fcn_O_pos(fea)
            return Ca_output, N_output, C_output, O_output
        else:
            return out_2