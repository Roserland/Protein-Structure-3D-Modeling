import imp
import os, sys, time
import json
import random
import argparse
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader,dataloader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score,jaccard_score,accuracy_score,confusion_matrix,roc_auc_score

from models.loss import WeightedCrossEntropyLoss, FocalLoss
from rmsd import rmsd_score
from models.UNet3D import UNet3D



class BackboneDiscriminator:
    def __init__(self, ckpt_path, device='cpu') -> None:
        self.ckpt_path = ckpt_path
        self.model = None
        self.device = device

        self.load_model()

    def load_model(self):
        self.model = torch.load(self.ckpt_path) # .to(self.device)

    
    def infer(self, img):
        self.model.eval()

        img = torch.reshape(img, (-1, 1, 64, 64, 64))
        logits = self.model(img)

        prob = torch.softmax(logits, dim=1)  # softmax on logits
        ones = torch.ones(img.shape)    #.to(device)
        zeros = torch.zeros(img.shape)  #.to(device)
        #predictions = torch.argmax(prob, dim=1).squeeze(dim=1)
        predictions = torch.where(prob[:, 1:2, :, :, :] > 0.51, ones, zeros).squeeze(dim=1)

        return predictions

    
    def infer_single(self, img):
        self.model.eval()
        # get confidence score, and calculate F1-Score, AUC, Precesion, Recall
        img = torch.reshape(img, (-1, 1, 64, 64, 64))
        pred_logits = self.model(img)

        # _targets = targets.permute(0, 2, 3, 4, 1)
        pred_logits = pred_logits.permute(0, 2, 3, 4, 1)
        scores, pred_labels = pred_logits.max(dim=-1)

        return pred_labels, scores
        



def get_predictions(pred_probs, using_sigmoid=True):
    """
    Args:
        pred_probs: torch.tensor, [bs, out_channels, H, W, D], eg. [32, 2, 32, 32, 32]
    """
    # [32, 32, 32, 32, 2]
    _preds = pred_probs.permute(0, 2, 3, 4, 1)
    num_classes = _preds.shape[-1]
    
    if using_sigmoid:
        cls_scores = _preds.sigmoid()
        scores = cls_scores.reshape(-1, num_classes)
        
        scores, indexs = cls_score.view(-1).topk()
        
    else:
        res = torch.softmax(logits, dim=-1)


def quant_scores(confidence, bin_size=0.001):
    _confidence_bins = confidence // bin_size
    _confidence_bins = np.array(_confidence_bins, dtype=np.uint32)

    res = _confidence_bins * bin_size
    return res


def time_counter():
    def deco_func(func):
        def wrapper(*args, **kwargs):
            curr_t = time.time()
            func(*args, **kwargs)
            print("[Time Using]: ", time.time() - curr_t)
        return wrapper
    return deco_func