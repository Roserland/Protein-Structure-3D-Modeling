"""
1. Construct Ca-Backbone Ground Truth, based on PoseEstimation Dataset
2. Will provide some loss definition here.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_recall_curve, roc_curve, auc
from scipy.ndimage.interpolation import zoom
from sklearn.model_selection import train_test_split
import random
import os
import os.path as osp
import pandas as pd
import numpy as np
import mrcfile
import sys
from collections import Counter

from sklearn.model_selection import train_test_split
import pandas as pd

sys.path.append('..')
from atom_pose_estimation.cube_loader import AminoAcidDataset, parse_offset, get_atoms_pos

# train_set = AminoAcidDataset(index_csv='../datas/split/train.csv', standard_size=[16, 16, 16], gt_type="heatmap")

def file_filter(pp_dir, suffix='backbone'):
    assert suffix in ['backbone', 'seg', '.txt']

    def _filter(file_name):
        return suffix in file_name

    file_list = os.listdir(pp_dir)
    new_list = filter(_filter, file_list)
    return list(new_list)

class CaBackboneSet:
    def __init__(self, 
                index_csv='../datas/backbone_reducer/train.csv',
                pp_dir='/mnt/data/zxy/amino-acid-detection/pp_dir/fzw_400_500', 
                label_mode="using_atom_O",
                logger=None) -> None:
        self.index_csv = index_csv
        self.pp_dir = pp_dir
        self.label_mode = label_mode

        self.load_file_list()
    
    def __getitem__(self, index: int):
        data_file = self.backbone_datas_file_list[index]
        label_file = self.backbone_labels_file_list[index]

        _data = np.load(osp.join(self.pp_dir, data_file))
        label = np.load(osp.join(self.pp_dir, label_file))

        if self.label_mode == "using_atom_O":
            #       O
            #       ||
            # Ca-N-C                                                           
            # which means, O atom is treated as the part of BackBone
            label[label == 2] = 1
        label[label != 1] = 0

        return _data, label
    
    def __len__(self):
        return len(self.backbone_labels_file_list)

    def load_file_list(self):
        df = pd.read_csv(self.index_csv)
        self.backbone_labels_file_list = df['file_name'].tolist()

        def get_data_file(file_name):
            tmp = file_name.split("_backbone")
            return tmp[0] + tmp[1]
        
        self.backbone_datas_file_list = list(
            map(get_data_file, self.backbone_labels_file_list))


def split_dataset(file_list, train_ratio=0.8, save_dir='../datas/backbone_reducer'):
    x_train, x_test = train_test_split(file_list, train_size=train_ratio)

    train_df = pd.DataFrame({"file_name": x_train})
    dst_path = os.path.join(save_dir, 'train.csv')
    train_df.to_csv(dst_path, index=None)

    test_df = pd.DataFrame({"file_name": x_test})
    dst_path = os.path.join(save_dir, 'test.csv')
    test_df.to_csv(dst_path, index=None)

if __name__ == '__main__':
    pp_dir = '/mnt/data/zxy/amino-acid-detection/pp_dir/fzw_400_500'
    # print(os.listdir(pp_dir))
    # train_set = CaBackboneSet(index_csv='../datas/split/train.csv', standard_size=[16, 16, 16], gt_type="heatmap")

    # rr = train_set.__getitem__(0)
    backbone_file_list = file_filter(pp_dir, suffix='backbone')
    # print(backbone_file_list)
    print(len(backbone_file_list))

    x_train, x_test = train_test_split(backbone_file_list)
    print(len(x_train))    
    print(len(x_test))
    # print(x_test)
    # split_dataset(backbone_file_list)

    train_set = CaBackboneSet(index_csv='../datas/backbone_reducer/train.csv')
    _data, label = train_set.__getitem__(0)
    print(Counter(label.reshape(-1)))