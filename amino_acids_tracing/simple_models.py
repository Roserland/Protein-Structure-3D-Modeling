"""
    Try some easy MachineLearning models to predict the afiinity matrix
"""

from turtle import forward
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from my_datasets import UniProtein


class MLP(nn.Module):
    def __init__(self, input_dim=24, hid_dim=128) -> None:
        super().__init__()
        
        self.layer_1 = nn.Sequential(nn.Linear(input_dim, hid_dim, bias=True), nn.ReLU())
        self.layer_2 = nn.Sequential(nn.Linear(hid_dim, hid_dim * 2, bias=True), nn.ReLU())
        self.layer_3 = nn.Sequential(nn.Linear(hid_dim * 2, int(hid_dim / 2), bias=True))
        self.layer_4 = nn.Sequential(nn.Linear(int(hid_dim / 2), 1, bias=True))

        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        out1 = self.layer_1(x)
        out2 = self.layer_2(self.dropout(out1))

        out3 = self.layer_3(self.dropout(out2))
        out4 = self.layer_4(self.dropout(out3))
        return out4

