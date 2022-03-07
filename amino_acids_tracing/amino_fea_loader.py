"""
    Amino acid feature vector dataset.
    The dataset will use the model trained in "Amino-Pose-Estimation" part.

"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from ..atom_pose_estimation.cube_loader import AminoAcidDataset, de_normalize
from ..atom_pose_estimation.inference import inference
from ..atom_pose_estimation.layers import HourGlass3DNet
import pandas as pd
import numpy as np
import mrcfile 
import os


AMINO_ACIDS = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
               'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']
AMINO_ACID_DICT = dict(zip(AMINO_ACIDS, range(20)))



def generate_data(model, index_csv='../datas/split/train.csv', save_dir='/mnt/data/zxy/stage3-amino-keypoint-vectors/'):
    model.eval()
    dataset = AminoAcidDataset(index_csv=index_csv, standard_size=[16, 16, 16],
                               gt_type='coords')
    dataset.set_mode(2)
    data_loader = DataLoader(dataset)
    df = pd.read_csv(index_csv)
    for i in range(len(df)):
        curr_records = df.iloc[i]
        _offset = curr_records["offset"]
        frag_id = curr_records['frag_id']
        mrc_path = curr_records['mrc_path']
        npy_file_path = mrc_path.replace('.mrc', '.npy')     
        amino_type = os.path.basename(mrc_path)[:3]
        amino_index = os.path.basename(mrc_path).strip('.mrc').split('_')[1]        # eg: */GLY_445.mrc
        cube_data_array = np.load(npy_file_path)

        npy_basename = os.path.basename(npy_file_path)
        coords_npy_file_path = npy_basename[:3] + '_coord' + npy_basename[3:]
        coords_npy_file_path = npy_file_path.replace(npy_basename, coords_npy_file_path)
        x1, x2, y1, y2, z1, z2, _ = np.load(coords_npy_file_path)
        upper_left_corner = [x1, y1, z1]
        lower_right_corner = [x2, y2, z2]

        # Inference, and prepare feature vector for a SINGLE amino acid
        Ca_output, N_output, C_output, O_output = model(cube_data_array)
        pred_Ca = de_normalize(Ca_output.detach().cpu().numpy(), upper_left_corner, lower_right_corner, _offset)
        pred_N  = de_normalize(N_output.detach().cpu().numpy(), upper_left_corner, lower_right_corner, _offset)
        pred_C  = de_normalize(C_output.detach().cpu().numpy(), upper_left_corner, lower_right_corner, _offset)
        pred_O  = de_normalize(O_output.detach().cpu().numpy(), upper_left_corner, lower_right_corner, _offset)
        # integrate type and 4 key point atoms positon
        fea_vec = np.zeros(32)      # 20  +  4 * 3
        fea_vec[-12:] = np.concatenate([pred_Ca, pred_N, pred_C, pred_O])
        fea_vec[AMINO_ACID_DICT[amino_type] - 1] = 1.0

        # save, 
        _save_dir = os.path.join(save_dir, frag_id)
        if not os.path.exists(_save_dir):
            os.makedirs(_save_dir)
        


def generate_simulation_data(pdb_index_csv):
    pass


        




class AminoFeatureDataset(Dataset):
    def __init__(self, index_csv, 
                 standard_size=[32, 32, 32],
                 gt_type=None,
                 zoom_type="diff"):
                 pass