#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py    
@Contact :   roserland@gmail.com
    Some common used functions
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/5 4:25 下午   fanzw      1.0         None
'''

# import
import os
import pandas as pd
import mrcfile
import numpy as np

def parse_atom(line):
    """Parse node's atom type from given 'line'"""
    return str.strip(line[12:16])


def parse_coordinates(line):
    """Parses node coordinates from given 'line'"""
    return [float(line[30:38]), float(line[38:46]), float(line[46:54])]


def parse_amino_acid(line):
    """Parses node amino acid from given 'line'"""
    return str.strip(line[17:20])


def parse_amino_acid_num(line):
    """Parses node amino acid num from given 'line'"""
    return str.strip(line[22:26])


def check_fragment_atom_nums(pdb_file_path):
    """
    1. Calculate atom nums within a single amino fragment
    2. Get atom counters
    :param pdb_file_path: amino fragment file, .pdb format
    """
    acid_num = os.path.basename(pdb_file_path)[:-4]
    origin_atoms_list = []
    atoms_name_list = []
    with open(pdb_file_path, 'r') as pdb_file:
        for line in pdb_file:
            atom_type_name = str.strip(line[77:])
            atom_name = str.strip(line[13:15])
            origin_atoms_list.append(atom_type_name)
            atoms_name_list.append(atom_name)
        amino_acid_type = parse_amino_acid(line)
    print(atoms_name_list)
    print(origin_atoms_list)
    print("Amino Type: ", amino_acid_type)


def change_voxel_size(mrc_file_path):
    mrc = mrcfile.open(mrc_file_path, "r+")
    # print(mrc.voxel_size)
    mrc.voxel_size = 1.0
    print(mrc.voxel_size)

    print(mrc.data.shape)


def gennerate_heat_map(data_array, atoms_pos):
    """
    generate heat map among 4 channels, with [Ca, N, C, O] order.
    : param data_array: 3D data array, loaded from mrc_file or its numpy array
    : param atoms_pos: List[Ca, N, C, O] 3D coordinates.
    """
    ca_heatmap = np.zeros()





if __name__ == '__main__':
    check_fragment_atom_nums(pdb_file_path='../test_data/pdb_fragments/3SJR/48.pdb')

    change_voxel_size(mrc_file_path='../test_data/mrc_fragments/3SJR/LEU_49.mrc')
