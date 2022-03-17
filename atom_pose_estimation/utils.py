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


def get_atoms_pos(pdb_file_path, aim_atom_types=None):
    """
        get particular atoms' 3D coordinates from .pdb file
        return: None or [[x1, y1, z1], [x2, y2, z2], ...]
    """
    if aim_atom_types is None:
        aim_atom_types = ["CA", "N", "C", "O"]
    res_dict = {}
    with open(pdb_file_path) as f:
        for line in f:
            atom_type = parse_atom(line)
            if atom_type in aim_atom_types:
                atom_coords = parse_coordinates(line)
                res_dict[atom_type] = atom_coords
            else:
                pass
    if len(res_dict) != len(aim_atom_types):
        print("Some bugs occur in {}".format(pdb_file_path))
        raise ValueError
    else:
        return [res_dict[key] for key in aim_atom_types]


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


class incre_std_avg():
    '''
    增量计算海量数据平均值和标准差,方差
    1.数据
    obj.avg为平均值
    obj.std为标准差
    obj.n为数据个数
    对象初始化时需要指定历史平均值,历史标准差和历史数据个数(初始数据集为空则可不填写)
    2.方法
    obj.incre_in_list()方法传入一个待计算的数据list,进行增量计算,获得新的avg,std和n(海量数据请循环使用该方法)
    obj.incre_in_value()方法传入一个待计算的新数据,进行增量计算,获得新的avg,std和n(海量数据请将每个新参数循环带入该方法)
    '''

    def __init__(self, h_avg=0, h_std=0, n=0):
        self.avg = h_avg
        self.std = h_std
        self.n = n

    def incre_in_list(self, new_list):
        avg_new = np.mean(new_list)
        incre_avg = (self.n*self.avg+len(new_list)*avg_new) / \
            (self.n+len(new_list))
        std_new = np.std(new_list)
        incre_std = np.sqrt((self.n*(self.std**2+(incre_avg-self.avg)**2)+len(new_list)
                                * (std_new**2+(incre_avg-avg_new)**2))/(self.n+len(new_list)))
        self.avg = incre_avg
        self.std = incre_std
        self.n += len(new_list)

    def incre_in_value(self, value):
        incre_avg = (self.n*self.avg+value)/(self.n+1)
        incre_std = np.sqrt((self.n*(self.std**2+(incre_avg-self.avg)
                                        ** 2)+(incre_avg-value)**2)/(self.n+1))
        self.avg = incre_avg
        self.std = incre_std
        self.n += 1




if __name__ == '__main__':
    check_fragment_atom_nums(pdb_file_path='../test_data/pdb_fragments/3SJR/48.pdb')

    change_voxel_size(mrc_file_path='../test_data/mrc_fragments/3SJR/LEU_49.mrc')
