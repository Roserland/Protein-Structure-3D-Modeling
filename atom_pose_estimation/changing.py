#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   changing.py    
@Contact :   roserland@gmail.com
    Get amino acid cubes, which includes:
        1. the amino_acid cube coordinates: [x1, x2, y1, y2, z1, z2]
        2. amino acid type, the index of variable "amino_acids"
        3. atom coordinates

    ATTENTION:
        1. 切割cube的时候, 保证使用的 mrc文件分辨率和pdb文件分辨率一致
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/12/28 5:06 下午   fanzw      1.0         None
'''

import argparse
import mrcfile
import os

# from configs import *
# from utils.pdb_utils import *  # delete
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

# params
box_size = 64           # box with padding
core_size = 50          # core part of protein_fragments

# all 20 amino-acids
amino_acids = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
               'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']


def get_amino_acid_coordinates(map_path, pdb_path, padding=2):
    """
    get amino acid cube coordinates
    :param map_path:    .mrc file path, used to get offsets
    :param pdb_path:    .pdb file path, get amino acid info
    :param padding:     padding size
    :return:            amino acid cube coordinates: List:[[x1, x2, y1, y2, z1, z2, amino_type_index]]
    """
    normalized_map = mrcfile.open(map_path, mode='r')
    origin = normalized_map.header.origin.item(0)  # offset
    # print("RS-TEST:  origin(offset) is : {}".format(origin))
    shape = normalized_map.data.shape
    amino_acid_coordinates = []
    with open(pdb_path, 'r') as pdb_file:
        amino_acid_num = 'None'
        amino_acid = 'None'
        x1, y1, z1, x2, y2, z2 = [shape[0] - 1, shape[1] - 1, shape[2] - 1, 0, 0, 0]
        for line in pdb_file:
            if line.startswith("ATOM"):
                if amino_acid_num == 'None':
                    amino_acid_num = parse_amino_acid_num(line)
                    amino_acid = parse_amino_acid(line)
                if parse_amino_acid_num(line) != amino_acid_num:
                    # padding is 2
                    if amino_acid in amino_acids and x1 - padding >= 0 and y1 - padding >= 0 and z1 - padding >= 0 \
                            and x2 + padding < shape[0] and y2 + padding < shape[1] and z2 + padding < shape[1]:
                        # 0 is background
                        amino_acid_coordinates.append([x1 - padding, x2 + padding, y1 - padding,
                                                       y2 + padding, z1 - padding, z2 + padding,
                                                       amino_acids.index(amino_acid) + 1])
                    amino_acid = parse_amino_acid(line)
                    amino_acid_num = parse_amino_acid_num(line)
                    x1, y1, z1, x2, y2, z2 = [shape[0] - 1, shape[1] - 1, shape[2] - 1, 0, 0, 0]

                coordinates = parse_coordinates(line)
                # pdb 是按 z, y, x 排列的
                # (x1, y1, z1): upper left;     (x2, y2, z2): lower right;
                z = int(coordinates[0] - origin[0]) * 2
                y = int(coordinates[1] - origin[1]) * 2
                x = int(coordinates[2] - origin[2]) * 2
                x1 = min(x1, x)
                x2 = max(x2, x)
                y1 = min(y1, y)
                y2 = max(y2, y)
                z1 = min(z1, z)

                z2 = max(z2, z)
    return amino_acid_coordinates


def crop_map_and_save_coordinates(map_path, pdb_path, output_path):
    amino_acid_coordinates = get_amino_acid_coordinates(map_path, pdb_path)
    print("amino_acid_coordinates:\n", amino_acid_coordinates)

    # paddle image, then process
    full_image = mrcfile.open(map_path, mode='r').data
    image_shape = np.shape(full_image)
    padded_image = np.zeros(
        (image_shape[0] + 2 * box_size, image_shape[1] + 2 * box_size, image_shape[2] + 2 * box_size), dtype=np.float64)
    padded_image[box_size: box_size + image_shape[0],
                 box_size: box_size + image_shape[1],
                 box_size: box_size + image_shape[2]] = full_image

    start_point = box_size - int((box_size - core_size) / 2)    # 64 - 7
    cur_x = start_point
    cur_y = start_point
    cur_z = start_point
    while cur_z + (box_size - core_size) / 2 < image_shape[2] + box_size:
        chunk = padded_image[cur_x:cur_x + box_size, cur_y:cur_y + box_size, cur_z:cur_z + box_size]
        chunk_coordinates = []
        r_x = cur_x - box_size
        r_y = cur_y - box_size
        r_z = cur_z - box_size
        for x1, x2, y1, y2, z1, z2, amino_acid_id in amino_acid_coordinates:
            if x1 >= r_x and x2 <= r_x + box_size and y1 >= r_y \
                    and y2 <= r_y + box_size and z1 >= r_z and z2 <= r_z + box_size:
                chunk_coordinates.append([x1 - r_x, x2 - r_x, y1 - r_y, y2 - r_y, z1 - r_z, z2 - r_z, amino_acid_id])
                print("R-TEST amino_acid_id: {}".format(amino_acid_id))
        if len(chunk_coordinates) > 0:
            # save chunk and coordinates
            chunk_name = pdb_path.split('/')[-1].split('.')[0] + '_' + str(int(r_x / core_size)) + '_' \
                         + str(int(r_y / core_size)) + '_' + str(int(r_z / core_size))
            np.save(os.path.join(output_path, chunk_name), chunk)
            with open(os.path.join(output_path, chunk_name + '.txt'), 'w') as label_file:
                for coordinate in chunk_coordinates:
                    label_file.write(','.join([str(i) for i in coordinate]) + '\n')
        cur_x += core_size
        if cur_x + (box_size - core_size) / 2 >= image_shape[0] + box_size:
            cur_y += core_size
            cur_x = start_point
            if cur_y + (box_size - core_size) / 2 >= image_shape[1] + box_size:
                cur_z += core_size
                cur_y = start_point
                cur_x = start_point


def split_protein_into_amino_acid(map_path, pdb_path, pdb_output_path, mrc_output_path, padding):
    """
    get amino acid cube coordinates while
    write amino_acid fragments into files, and store them
    :param map_path:    .mrc file path, used to get offsets
    :param pdb_path:    .pdb file path, get amino acid info
    :param padding:     padding size
    :param pdb_output_path: dst dir, save pdb fragments, named with PDB name, then amino acids num
    :param mrc_output_path: dst dir, save mrc fragments, named with PDB name, then amino acids num
    :return:            amino acid cube coordinates: List:[[x1, x2, y1, y2, z1, z2, amino_type_index]]
    """
    # prepare protein fragments directory
    pdb_id = os.path.basename(pdb_path)[:-4]
    pdb_fragments_dir = os.path.join(pdb_output_path, pdb_id)
    if not os.path.exists(pdb_fragments_dir):
        os.makedirs(pdb_fragments_dir)
    mrc_fragments_dir = os.path.join(mrc_output_path, pdb_id)
    if not os.path.exists(mrc_fragments_dir):
        os.makedirs(mrc_fragments_dir)

    def write_fragment(lines_list, file_path):
        with open(file_path, 'w') as f:
            f.writelines(lines_list)

    # paddle image, then process
    normalized_map = mrcfile.open(map_path, mode='r')
    origin = normalized_map.header.origin.item(0)  # offset
    full_image = normalized_map.data

    # print("RS-TEST:  origin(offset) is : {}".format(origin))
    shape = normalized_map.data.shape
    amino_acid_coordinates = []
    with open(pdb_path, 'r') as pdb_file:
        amino_acid_num = 'None'
        amino_acid = 'None'
        x1, y1, z1, x2, y2, z2 = [shape[0] - 1, shape[1] - 1, shape[2] - 1, 0, 0, 0]

        lines_list = []
        for line in pdb_file:
            if line.startswith("ATOM"):
                if amino_acid_num == 'None':
                    amino_acid_num = parse_amino_acid_num(line)
                    amino_acid = parse_amino_acid(line)
                    lines_list.append(line)
                if parse_amino_acid_num(line) != amino_acid_num:
                    # new amino acid atoms
                    # write the old fragment
                    fragment_path = os.path.join(pdb_fragments_dir, str(amino_acid_num) + '.pdb')       # write sub-pdb
                    write_fragment(lines_list, file_path=fragment_path)
                    lines_list = [line]
                    # padding is 2
                    if amino_acid in amino_acids:
                        if x1 - padding >= 0 and y1 - padding >= 0 and z1 - padding >= 0 \
                                and x2 + padding < shape[0] and y2 + padding < shape[1] and z2 + padding < shape[1]:
                            # 0 is background
                            amino_acid_coordinates.append([x1 - padding, x2 + padding, y1 - padding,
                                                        y2 + padding, z1 - padding, z2 + padding,
                                                        amino_acids.index(amino_acid) + 1])
                            # TODO: add fragments splitting
                            cube = full_image[x1 - padding: x2 + padding,
                                              y1 - padding: y2 + padding,
                                              z1 - padding: z2 + padding,]
                            cube_path = os.path.join(mrc_fragments_dir, amino_acid + '_' + str(amino_acid_num) + '.npy')
                            np.save(cube_path, cube)

                            if '.rebuilt' in mrc_fragments_dir:
                                cube_coords_path = os.path.join(mrc_fragments_dir[:-8],
                                                                amino_acid + '_coord_' + str(amino_acid_num) + '.npy')
                            else:
                                cube_coords_path = os.path.join(mrc_fragments_dir,
                                                                amino_acid + '_coord_' + str(amino_acid_num) + '.npy')
                            tmp_mrc_file_name = os.path.join(mrc_fragments_dir,
                                                             amino_acid + '_' + str(amino_acid_num) + '.mrc')
                            # print([x1 - padding, x2 + padding,
                            #          y1 - padding, y2 + padding,
                            #          z1 - padding, z2 + padding,
                            #          amino_acids.index(amino_acid) + 1])
                            print(cube_coords_path)

                            # save coords
                            np.save(cube_coords_path,
                                    [x1 - padding, x2 + padding,
                                     y1 - padding, y2 + padding,
                                     z1 - padding, z2 + padding,
                                     amino_acids.index(amino_acid) + 1])
                            with mrcfile.new(tmp_mrc_file_name) as tmp_mrc:                   # write sub-mrc
                                tmp_mrc.set_data(cube)
                        else:
                            print("amino acid num {} in protein {} is at the boundary".format(amino_acids.index(amino_acid) + 1, pdb_id))
                    amino_acid = parse_amino_acid(line)
                    amino_acid_num = parse_amino_acid_num(line)
                    x1, y1, z1, x2, y2, z2 = [shape[0] - 1, shape[1] - 1, shape[2] - 1, 0, 0, 0]
                else:
                    lines_list.append(line)

                coordinates = parse_coordinates(line)
                # pdb 是按 z, y, x 排列的
                # (x1, y1, z1): upper left;     (x2, y2, z2): lower right;
                # 已经减去了offset
                z = int(coordinates[0] - origin[0]) * 2
                y = int(coordinates[1] - origin[1]) * 2
                x = int(coordinates[2] - origin[2]) * 2
                x1 = min(x1, x)
                x2 = max(x2, x)
                y1 = min(y1, y)
                y2 = max(y2, y)
                z1 = min(z1, z)
                z2 = max(z2, z)
    return amino_acid_coordinates


def test():
    mrc_file_path = '../test_data/test/3SJR.mrc'
    pdb_file_path = '../test_data/test/3SJR.pdb'

    # crop_map_and_save_coordinates(mrc_file_path, pdb_file_path, output_path='../test_data/coords/')

    # split_protein_into_amino_acid(mrc_file_path, pdb_file_path, pdb_output_path='../test_data/pdb_fragments/',
    #                               mrc_output_path='../test_data/mrc_fragments', padding=2)

    split_protein_into_amino_acid(mrc_file_path, pdb_file_path,
                                  pdb_output_path='{}/test_data/pdb_fragments/'.format(root_dir),
                                  mrc_output_path='{}/test_data/mrc_fragments/'.format(root_dir), padding=2)


def get_mrc_offset(path):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--EMdata_dir', type=str, required=True)
    # parser.add_argument('--pp_dir', type=str, required=True)
    parser.add_argument('--EMdata_dir', type=str, required=False)
    parser.add_argument('--pp_dir', type=str, required=False)
    args = parser.parse_args()

    box_size = box_size
    core_size = core_size

    # os.makedirs(args.pp_dir, exist_ok=True)
    args.EMdata_dir = "/Volumes/RS/amino-acid-detection/EMdata_dir/400_500"
    args.EMdata_dir = "/mnt/data/zxy/amino-acid-detection/EMdata_dir/400_500"
    data_root_dir = "/mnt/data/zxy/amino-acid-detection//test_data"
    for pdb_id in os.listdir(args.EMdata_dir):
        map_path = os.path.join(args.EMdata_dir, pdb_id, 'simulation/normalized_map.mrc')
        pdb_path = os.path.join(args.EMdata_dir, pdb_id, 'simulation/{}.rebuilt.pdb'.format(pdb_id))
        if os.path.exists(map_path) and os.path.exists(pdb_path):
            # crop_map_and_save_coordinates(map_path, pdb_path, args.pp_dir)
            split_protein_into_amino_acid(map_path, pdb_path,
                                          pdb_output_path='{}/pdb_fragments/'.format(data_root_dir),
                                          mrc_output_path='{}/mrc_fragments/'.format(data_root_dir), padding=2)
            print('finish {}'.format(pdb_id))

    # test()



