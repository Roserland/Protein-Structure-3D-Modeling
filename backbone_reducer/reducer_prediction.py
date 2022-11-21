import torch
import mrcfile
import os, json
import os.path as osp
import h5py
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader
from collections import deque
from models.UNet3D import UNet3D
import math
from utils.map_splitter import reconstruct_map, create_manifest
from rmsd import rmsd_score,load_data,probable_ca_mask,distance
from postprocessing.pdb_reader_writer import *
import argparse
import pandas as pd
from collections import Counter

from prediction import prediction_and_visualization_backbone_reducer


if __name__ == '__main__':
    """
    Usage :
    python prediction.py    --dataset_path /mnt/data/Storage4/mmy/CryoEM_0112_train  \
                            --reducer_model_path ./checkpoints/0112_0920_atom_simulation_checkpoint_epoch_29   \
                            --type simulation   \
                            --overwrite
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--reducer_model_path', type=str, required=True)
    parser.add_argument('--type', type=str, required=True)
    parser.add_argument('--overwrite', action='store_const', const=True, default=False)
    parser.add_argument('--output_channel', default=2, type=int)

    args = parser.parse_args()
    

    amino_acids = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
                   'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']


    backbone_reducer_model_path = args.reducer_model_path

    test_proteins_dir = "/mnt/data/zxy/amino-acid-detection/pp_dir/whole_protein"
    test_dir = test_proteins_dir
    pid_list = os.listdir(test_proteins_dir)
    for pid in pid_list:
        print("processing {}".format(pid))
        prediction_and_visualization_backbone_reducer(backbone_reducer_model_path, args, pid, 
                                                      test_proteins_dir)
        # atom_save_path = os.path.join(test_dir, pid, args.type, 'atom_prediction.mrc')
        # acid_save_path = os.path.join(test_dir, pid, args.type, 'amino_acid_prediction.mrc')

        # gt_pdb_file_path = os.path.join(test_dir, pid, pid + '_ca.pdb')
        # pred_pdb_file_path = os.path.join(test_dir, pid)
        # rmsd_score(atom_save_path, acid_save_path, gt_pdb_file_path, pid, pred_pdb_file_path)
        # data = pd.read_excel('/home/fzw/Cryo-EM/Protein-Structure-3D-Modeling/ca_filter/tmp/{}_outputresults.xls'.format(pid))
        # df = df.append(data[0:1])