"""
@Roserland  2022-10-10

1. Construct Ca-Backbone Ground Truth, based on PoseEstimation Dataset
2. Will provide some loss definition here.

All the data are 827 protein samples: If no more datas...
    Train: 640 samples
    Test:  160 samples
    Valid: 27  samples

Each sample is a complete Protein File, with the size of (like) 236 ** 3;
    After pre-processiong, each sample will gengerate several chunks, and 
    corresponding labels: detecion-gt, seg-gt, backbone-seg-get, pose-estimation gt

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
from utils.decoder import time_counter

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
                out_channels=2,
                logger=None) -> None:
        self.index_csv = index_csv
        self.pp_dir = pp_dir
        self.label_mode = label_mode
        self.out_channels = out_channels

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
        if self.out_channels == 2:
            label[label != 1] = 0
        elif self.out_channels == 3:
            label[label == 3] = 2
        else:
            raise ValueError

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


@time_counter()
def empty_chunks_filter(index_csv='../datas/backbone_reducer/train.csv', pp_dir='/mnt/data/zxy/amino-acid-detection/pp_dir/fzw_400_500'):
    df = pd.read_csv(index_csv)
    backbone_labels_file_list = df['file_name'].tolist()
    print("Original Chunks Num is: ", len(backbone_labels_file_list))

    # do filtering
    # nums_2000:  8097
    # nums_1600:  7067
    # nums_1200:  5904
    # nums_1000:  5200
    # nums_800:   4321
    # nums_600:   4321
    # nums_0:     0
    new_name_lsit = []
    nums_2000 = 0
    nums_1600 = 0
    nums_1200 = 0
    nums_1000 = 0
    nums_800 = 0
    nums_600 = 0
    nums_0 = 0
    for i in range(len(backbone_file_list)):
        curr_name = backbone_file_list[i]
        print(curr_name)
        file_path = os.path.join(pp_dir, curr_name)

        labels = np.load(file_path)
        counter = Counter(labels.reshape(-1))

        print(counter)

        counter_nums = counter[1] + counter[2]
        if counter_nums < 2000:
            nums_2000 += 1
        if counter_nums < 1600:
            nums_1600 += 1
        if counter_nums < 1200:
            nums_1200 += 1
        if counter_nums < 1000:
            nums_1000 += 1
        if counter_nums < 800:
            nums_800 += 1
        if counter_nums < 800:
            nums_600 += 1
        if counter_nums <= 0:
            nums_0 += 1
    
    print("nums_2000: ", nums_2000)
    print("nums_1600: ", nums_1600)
    print("nums_1200: ", nums_1200)
    print("nums_1000: ", nums_1000)
    print("nums_800: ", nums_800)
    print("nums_600: ", nums_600)
    print("nums_0: ", nums_0)

if __name__ == '__main__':
    pp_dir = '/mnt/data/zxy/amino-acid-detection/pp_dir/fzw_400_500'
    # print(os.listdir(pp_dir))
    # train_set = CaBackboneSet(index_csv='../datas/split/train.csv', standard_size=[16, 16, 16], gt_type="heatmap")

    # rr = train_set.__getitem__(0)
    backbone_file_list = file_filter(pp_dir, suffix='backbone')
    # print(backbone_file_list)
    print(len(backbone_file_list))

    x_train, x_test = train_test_split(backbone_file_list, train_size=0.8)
    print(len(x_train))    
    print(len(x_test))
    # print(x_test)
    # split_dataset(backbone_file_list)

    # train_set = CaBackboneSet(index_csv='../datas/backbone_reducer/train.csv')
    # _data, label = train_set.__getitem__(0)
    # print(Counter(label.reshape(-1)))

    empty_chunks_filter()