#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   usual_test.py    
@Contact :   roserland@gmail.com

Test file,  test some functions

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/9 10:51 下午   fanzw      1.0         None
'''

# import lib
import os
import pandas as pd
import numpy as np
import mrcfile
import matplotlib.pyplot as plt
from cube_loader import *

# mrc_fragment_dirs = '/Volumes/RS/amino-acid-detection/test_data/mrc_fragments'
# pdb_fragment_dirs = '/Volumes/RS/amino-acid-detection/test_data/pdb_fragments'

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


def get_atoms_pos(pdb_file, aim_atom_types=["CA", "N", "C", "O"]):
    """
        get particular atoms' 3D coordinates from .pdb file
        return: None or [[x1, y1, z1], [x2, y2, z2], ...]
    """
    res_dict = {}
    with open(pdb_file) as f:
        for line in f:
            atom_type = parse_atom(line)
            if atom_type in aim_atom_types:
                atom_coords = parse_coordinates(line)
                res_dict[atom_type] = atom_coords
            else:
                pass
    if len(res_dict) != len(aim_atom_types):
        return None
    else:
        return [res_dict[key] for key in aim_atom_types]



def generate_index_csv(files_root_dir, suffix="pdb"):
    _protein_ids_list = os.listdir(files_root_dir)
    if len(_protein_ids_list[0]) > 4:
        protein_ids_list = _protein_ids_list[1:]
    else:
        protein_ids_list = _protein_ids_list
    print(protein_ids_list)

    frag_name_list = []
    frag_path_list = []
    for p_id in protein_ids_list:
        p_dir = os.path.join(files_root_dir, p_id)
        fragment_list = os.listdir(p_dir)
        for file_name in fragment_list:
            if file_name[-3:] != suffix:
                pass
            else:
                # if '.mrc' in file_name:
                _file_name = file_name.split('_')[-1].strip()
                # else:
                #     _file_name = file_name
                index_name = '_'.join([p_id, _file_name])
                frag_name_list.append(index_name)
                frag_path_list.append(os.path.join(p_dir, file_name))
    # need sorting
    df = pd.DataFrame({"frag_id": frag_name_list, "path": frag_path_list})
    df = df.sort_values(by="frag_id")
    csv_name = "{}_index.csv".format(suffix)
    df.to_csv('./{}'.format(csv_name), index=None)
    return protein_ids_list


if __name__ == '__main__':
    # *********************  Part-1 Begin  *********************
    # mrc_names = generate_index_csv(mrc_fragment_dirs, suffix='mrc')
    # pdb_names = generate_index_csv(pdb_fragment_dirs, suffix='pdb')
    # print(mrc_names == pdb_names)
    # df = {"protein_id": mrc_names}
    # df = pd.DataFrame(df)
    # df.to_csv('./protein_id.csv', index=None)

    # EMdata_dir = '/Volumes/RS/amino-acid-detection/EMdata_dir/400_500/'
    # offset_dict = {}
    # for pdb_id in os.listdir(EMdata_dir):
    #     if len(pdb_id) > 4:
    #         continue
    #     map_path = os.path.join(EMdata_dir, pdb_id, 'simulation/normalized_map.mrc')
    #     # pdb_path = os.path.join(EMdata_dir, pdb_id, 'simulation/{}.rebuilt.pdb'.format(pdb_id))
    #     normalized_map = mrcfile.open(map_path, mode='r')
    #     origin = normalized_map.header.origin.item(0)  # offset
    #     _offset_str = str(round(origin[0], 5)) + ';' + str(round(origin[1], 5)) + ';' + str(round(origin[2], 5))
    #     # print(origin)
    #     # print(_offset_str)
    #     offset_dict[pdb_id] = _offset_str
    # #
    # mrc_index_df = pd.read_csv('/Users/fanzw/PycharmProjects/3D-Detection/cryoem-amino-acid-detection/3d_posetimtation/mrc_index.csv')
    # print(mrc_index_df.head(5))

    # get offset
    # def get_protein_id(_str):
    #     return _str[:4]
    # protein_id = mrc_index_df['frag_id'].map(get_protein_id)
    # mrc_index_df["offset"] = protein_id.map(offset_dict)
    # # print(mrc_index_df.head(5))
    # mrc_index_df.to_csv('/Users/fanzw/PycharmProjects/3D-Detection/cryoem-amino-acid-detection/3d_posetimtation/mrc_index.csv', index=None)

    # *********************  Part-1 Finished  *********************

#
    # shape_list = []
    # for map_path in mrc_index_df['path'].tolist():
    #     normalized_map = mrcfile.open(map_path, mode='r')
    #     shape = list(normalized_map.data.shape)
    #     print(shape)
    #     shape_list.append(shape)
    # shape_list = np.array(shape_list)
    #
    # print(np.max(shape_list, axis=0))       # [36 38 60]
    # np.save('cube_shapes.npy', shape_list)

    # shape_list_array = np.load('cube_shapes.npy')
    # print(shape_list_array.shape)
    # x = shape_list_array[:, 0]
    # y = shape_list_array[:, 1]
    # z = shape_list_array[:, 2]
    # plt.scatter(list(range(len(x))), x, color='red', label="x", s=1, marker='*')
    # plt.scatter(list(range(len(x))), y, color='y', label="y", s=1, marker='+')
    # plt.scatter(list(range(len(x))), z, color='b', label="z", s=1, marker='o')
    # plt.savefig('./length_distribution.jpg')

    test_set = AminoAcidDataset(index_csv='../datas/split/test.csv', )
    print("length of origin test set {}".format(test_set.__len__()))
    for i in range(20):
        print(test_set.mrc_cube_path[i])

    print("\n\n\n")
    uni_test_set = uniAminoTypeDataset(amino_type="ASP", index_csv='../datas/split/test.csv')
    print("length of uniType test set {}".format(uni_test_set.__len__()))
    for i in range(20):
        print(uni_test_set.mrc_cube_path[i])
        # print(uni_test_set.pdb_cube_path[i])
