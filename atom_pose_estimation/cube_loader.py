#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   cube_loader.py    
@Contact :   roserland@gmail.com

1. Data loader file, will prepare 'Ca, N, CO' atoms' position
2. Will provide some loss definition here.

TODO:
    1. Change the rescaling method: for example, non-equal at all axis
    2. No need to zoom the cube to a fixed size:
        2.1 if larger, trunk
        2.2 if smaller, zoom then padding.
    3. Some unvalid data need to be filtered.
    4. add random padding to the sub-pdb-cube
/Volumes/RS/amino-acid-detection/test_data/mrc_fragments/6XAF/ASP_1458.mrc /Volumes/RS/amino-acid-detection/test_data/pdb_fragments/6XAF/1458.pdb
/6P3U/128.pdb
/7B1Q/46.pdb
/6QR1/47.pdb
/6XAF/1458.pdb
/7B1E/46.pdb
/6QQU/76.pdb
/6VV3/350.pdb
/7B1Q/48.pdb
/7NH5/19.pdb
/7B1E/48.pdb
/6XQ6/98.pdb
/7B1P/48.pdb
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/9 10:35 下午   fanzw      1.0         None
'''

# import lib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_recall_curve, roc_curve, auc
from scipy.ndimage.interpolation import zoom
from sklearn.model_selection import train_test_split
import random
import os
import pandas as pd
import numpy as np
import mrcfile


# coordinates loss
# A. MSE loss
# B. L1 loss
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


def parse_offset(off_str):
    x, y, z = off_str.split(';')
    return float(x.strip()), float(y.strip()), float(z.strip())


def de_normalize(related_pos, upper_left_corner, lower_right_corner, mrc_offset):
    """
    return a real pos of an atom
    :param cube_coords: [x1, x2, y1, y2, z1, z2]
    """
    # [x1, x2, y1, y2, z1, z2] = cube_coords
    # upper_left_corner = [x1, y1, z1]
    # lower_right_corner = [x2, y2, z2]
    cube_size = np.array(lower_right_corner) - np.array(upper_left_corner)
    real_pos = np.array(related_pos) * cube_size + upper_left_corner
    x, y, z = real_pos
    _x = x/2 + mrc_offset[2]
    _y = y/2 + mrc_offset[1]
    _z = z/2 + mrc_offset[0]
    return np.array([_z, _y, _x])


def batch_de_normalize(related_pos_arr, upper_left_corner_arr, lower_right_corner_arr, mrc_offset_arr):
    length = related_pos_arr.shape[0]

    res = []
    for i in range(length):
        _coord = de_normalize(related_pos_arr[i], upper_left_corner_arr[i], lower_right_corner_arr[i], mrc_offset_arr[i])
        res.append(_coord)
    return res


def rescale(cube_array, aim_size=[32, 32, 32]):
    """
    rescale a cube to a standard size, and return the rescaled coordinates
    """
    "scipy.ndiamge.interpolation.zoom"
    pass


def calculate_spatial_angle(aim_cube_pdb, base_cube_pdb):
    """
    Calculate a spatial offset and rotation angle from a 'base_cube', which is used among all amino-acid-cube which
    has the same amino-acid-type.
    : return: a rotation matrix
    """
    pass


class HeatmapLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction="mean") -> None:
        super(HeatmapLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction= reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        """
        logits and label have same shape, and the label's data type is long
        args:
            logits: tensor of shape (N, ...)
            label:  tensor of shape (N, ...)
        """
        logits = logits.float()

        with torch.no_grad():
            aplha = torch.empty_like(logits).fill_(1 - self.alpha)
            aplha[label == 1] = self.alpha      # label > 0.2 ?
            # aplha[label >  0] = self.alpha      # label > 0.2 ?
        
        probs = torch.sigmoid(logits)





def heatmap_loss():
    '''
    modified from focal loss with CE
    '''
    


def myLoss(tag='L1'):
    if tag == "L1":
        pass
    elif tag == 'L2':
        pass
    else:
        pass


class HeatmapGenerator3D:
    def __init__(self, sigma, kernel_size=3, offset=0, k_shape=None) -> None:
        self.sigma = sigma
        self.kz = kernel_size
        self.offset = offset
        assert self.kz % 2 == 1
        if k_shape is None:
            self.k_shape = [kernel_size] * 3 
        else:
            self.k_shape = k_shape
        
        self.kernel = self.generate_gaussian_kernel()
    

    def heatmap3D(self, data_array, atoms_pos):
        """
        : data_array: 3D numpy array, loaded from .mrcfile  or .npy data
        : atoms_pos: List: [Ca:[x, y, z], N, C, O], the item is a 3D  coordinates [x, y, z] , relative coords, x is in [0, 1]
        """
        h, w, c = data_array.shape
        pos_nums  = len(atoms_pos)
        kx, ky, kz = self.k_shape
        heatmaps = np.zeros([pos_nums, h + kx - 1, w + ky - 1, c + kz - 1])
        for i, heatmap in enumerate(heatmaps):
            x, y, z = int(round(h * atoms_pos[i][0])), int(round(w * atoms_pos[i][1])),  int(round(c * atoms_pos[i][2]))
            heatmap[x - kx//2 : x + kx // 2 + 1, 
                    y - ky//2 : y + ky // 2 + 1, 
                    z - kz//2 : z + kz // 2 + 1, ] = self.kernel
        
        heatmaps = heatmaps[:,  kx // 2 : h + kx // 2, ky // 2 : w + ky // 2, kz // 2 : h + kz // 2, ]
        return heatmaps


    def generate_gaussian_kernel(self):
        """
            just standard gaussian now
        """
        kernel = np.zeros([self.kz, self.kz, self.kz,])
        center = np.array([self.kz // 2, self.kz // 2, self.kz // 2])

        for i in range(self.kz):
            for j in range(self.kz):
                for q in range(self.kz):
                    pos_diff = np.array([i, j, q]) - center - self.offset
                    kernel[i, j, q] = np.exp(-(pos_diff ** 2).sum() / (2 * self.sigma ** 2))
        # kernel /= (self.sigma * np.sqrt(2 * np.pi))
        return kernel
    
    
    def get_heatmap_pos(self, data_array, relate=True, thres=0):
        data_array[data_array < thres] = 0
        h, w, c = data_array.shape
        
        _data = data_array.reshape(-1)
        _max = np.max(_data)
        idx = np.argmax(_data)

        for i in range(h):
            for j in range(w):
                for k in range(c):
                    if _max == data_array[i, j, k]:
                        x, y, z = i, j, k
        # weight_x = []        
        # for i in range(h):
        #     _arr = data_array[i, :, :]
        #     weight_x.append(_arr.sum())
        # x = (np.array(weight_x) * np.arange(h)).sum()  / (np.array(weight_x).sum() + 1e-8)

        # weight_y = []        
        # for i in range(w):
        #     _arr = data_array[:, i, :]
        #     weight_y.append(_arr.sum())
        # y = (np.array(weight_y) * np.arange(h)).sum()  / (np.array(weight_y).sum() + 1e-8)

        # weight_x = []        
        # for i in range(c):
        #     _arr = data_array[:, :, i]
        #     weight_x.append(_arr.sum())
        # z = (np.array(weight_x) * np.arange(h)).sum()  / (np.array(weight_x).sum() + 1e-8)

        if relate:
            return np.array([x/h, y/w, z/c])
        return np.array([x, y, z])
    

    def get_maps_pos(self, data_array, thres=0):
        N, h, w, c = data_array.shape
        res = []
        for i in range(N):
            coords = self.get_heatmap_pos(data_array[i])
            res.append(list(coords))
        return np.array(res) 

        
        # kernel = 1 / self.kernel
        # kx, ky, kz = kernel.shape
        # h, w, c = data_array.shape
        # # assumption: equal length among each dimension
        # assert kx == ky

        # pad = kx // 2
        # new_map = np.zeros([h + pad*2, w + pad*2, c + pad*2, ])
        # res = np.zeros(h, w, c)
        # for i in range(pad, h + pad):
        #     for j in range(pad, w + pad):
        #         for k in range(pad, c + pad):
        #             pass



def generate_index_csv(files_root_dir, suffix="pdb", output_dir='./datas/'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
    df.to_csv(os.path.join(output_dir, csv_name), index=None)
    return protein_ids_list, os.path.join(output_dir, csv_name)


def generate_index_and_split_data(EMdata_dir, mrc_fragment_dirs, pdb_fragment_dirs, output_dir='./datas/'):
    """
    """
    mrc_names, mrc_index_path = generate_index_csv(mrc_fragment_dirs, suffix='mrc', output_dir=output_dir)
    pdb_names, pdb_index_path = generate_index_csv(pdb_fragment_dirs, suffix='pdb', output_dir=output_dir)

    # EMdata_dir = '/Volumes/RS/amino-acid-detection/EMdata_dir/400_500/'
    offset_dict = {}
    for pdb_id in os.listdir(EMdata_dir):
        if len(pdb_id) > 4:
            continue
        map_path = os.path.join(EMdata_dir, pdb_id, 'simulation/normalized_map.mrc')
        normalized_map = mrcfile.open(map_path, mode='r')
        origin = normalized_map.header.origin.item(0)  # offset
        _offset_str = str(round(origin[0], 5)) + ';' + str(round(origin[1], 5)) + ';' + str(round(origin[2], 5))
        offset_dict[pdb_id] = _offset_str

    mrc_index_df = pd.read_csv(mrc_index_path)
    print(mrc_index_df.head(5))

    # get offset
    def get_protein_id(_str):
        return _str[:4]
    # add offset
    protein_id = mrc_index_df['frag_id'].map(get_protein_id)
    mrc_index_df["offset"] = protein_id.map(offset_dict)
    mrc_index_df.to_csv(mrc_index_path, index=None)


def split_data(pdb_index_csv, mrc_index_csv, output_dir='./split/', val_split=0.2):
    """
    Split data into 'Train, Test, Valid' set
    """
    def drop_suffix(name_str):
        return name_str[:-4]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    _pdb_df = pd.read_csv(pdb_index_csv)
    _pdb_df["frag_id"] = _pdb_df["frag_id"].map(drop_suffix)
    _pdb_df.columns = ['frag_id', 'pdb_path']
    _mrc_df = pd.read_csv(mrc_index_csv)
    _mrc_df["frag_id"] = _mrc_df["frag_id"].map(drop_suffix)
    _mrc_df.columns = ['frag_id', 'mrc_path', 'offset']
    print("all sub-pdb nums:", len(_pdb_df), '\n', "all sub-mrc nums", len(_mrc_df))
    new_df = pd.merge(_mrc_df, _pdb_df, how='inner')
    print(new_df.head())
    print(len(new_df))

    train_idx, val_idx = train_test_split(list(range(len(new_df))), test_size=val_split)
    _train_df = new_df.iloc[train_idx].reset_index(drop=True)
    valid_df = new_df.loc[val_idx].reset_index(drop=True)

    train_idx, test_idx = train_test_split(list(range(len(_train_df))), test_size=0.2)
    train_df = _train_df.iloc[train_idx].reset_index(drop=True)
    test_df = _train_df.iloc[test_idx].reset_index(drop=True)

    # save
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=None)
    valid_df.to_csv(os.path.join(output_dir, "valid.csv"), index=None)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=None)

    print("Train sets len:{} \t Valid sets len: {}\t Test sets len: {}".format(len(train_df), len(valid_df),
                                                                               len(test_df)))


class AminoAcidDataset(Dataset):
    def __init__(self, index_csv, 
                 standard_size=[32, 32, 32],
                 gt_type=None,
                 zoom_type="diff"):
        df = pd.read_csv(index_csv)
        self.pdb_cube_path = df['pdb_path'].tolist()
        self.mrc_cube_path = df['mrc_path'].tolist()
        self.offset_list = df['offset'].tolist()
        self.gt_type = gt_type
        print(self.gt_type, type(self.gt_type))
        self.standard_size = standard_size
        self.zoom_type = zoom_type
        self.mode = 1
        assert len(self.pdb_cube_path) == len(self.mrc_cube_path)

        self.heatmap_generator = HeatmapGenerator3D(sigma=1.0, kernel_size=3)

    def normalize_coords(self, coords, offset, upper_left, lower_right, if_norm=True, padding=2):
        """
        Coordinates normalization: Transfer coordinates to 0-1 interval
        : coords: [x, y, z]
        : offset: original .mrc offset --> [x, y, z]
        : cube size: shape, [x, y, z]
        """
        # TODO: find atom_pos among a selected cube
        offset_xyz = np.array([offset[2], offset[1], offset[0],])
        # _coords = np.array(coords) - np.array(offset_xyz)
        _cube_size = np.array(lower_right) - np.array(upper_left)
        real_upper_left = (np.array(upper_left) + padding) / 2 + offset_xyz
        real_lower_right = (np.array(lower_right) - padding) / 2 + offset_xyz
        # _cube_size = np.array(cube_size)
        z = float(coords[0] - offset[0]) * 2
        y = float(coords[1] - offset[1]) * 2
        x = float(coords[2] - offset[2]) * 2
        _coords = np.array([x, y, z])
        # print("Relate to original offset: \t", _coords)
        _coords = np.round(np.array(np.array(_coords) - np.array(upper_left)), 3)
        if (_coords > _cube_size).sum() > 0:
            print(_coords, _cube_size)
            print("coordinates is out of the cube boundary")
            raise ValueError
        
        norm_coords = _coords / _cube_size      
        
        return np.array(_coords), np.array(norm_coords)


    def rescale(self, cube_array):
        """
        rescale a cube to a standard size, and return the rescaled coordinates
        """
        _aim_size = self.standard_size
        a, b, c = cube_array.shape
        # TODO: Check if the
        scale = (self.standard_size[0] / a,
                 self.standard_size[1] / b,
                 self.standard_size[2] / c,)
        if self.zoom_type =='same':
            # all dimensions zoomed at a same factor, so the output may not be the aim_size
            # need padding
            min_factor = min(scale)
            _rescaled = zoom(cube_array, min_factor)

            x, y, z = _rescaled.shape
            res = np.zeros([self.standard_size])
            res[(self.standard_size[0] - x) // 2:(self.standard_size[0] - x) // 2 + x,
                (self.standard_size[0] - y) // 2:(self.standard_size[0] - y) // 2 + y,
                (self.standard_size[0] - z) // 2:(self.standard_size[0] - z) // 2 + z] = _rescaled
            return res
        else:
            # all dimensions zoomed with different factors, so the output is the aim_size
            return zoom(cube_array, scale)

    
    def set_mode(self, mode):
        """
        mode == 1: train
        mode == 2: test / valid
        """
        self.mode = mode
    
    def __getitem__(self, item):
        # TODO:
        #  1. Need to take 'offset' into account, for particular pos in a single cube --> done
        #  2. Check the cube size, if it's suitable feed into model                   --> done
        #     2.1 There need to set a format size of cube, padding for those under-sized cube  --> done
        #     2.2 For those extra large cubes, it may rescale, then restore the coordinates --> No need
        #     2.3 Normalize the coordinates to 0-1 interval, after substracting offset  --> done
        pdb_file_path = self.pdb_cube_path[item]
        mrc_file_path = self.mrc_cube_path[item]
        _offset = parse_offset(self.offset_list[item])
        npy_file_path = mrc_file_path.replace('.mrc', '.npy')
        npy_basename = os.path.basename(npy_file_path)
        coords_npy_file_path = npy_basename[:3] + '_coord' + npy_basename[3:]
        coords_npy_file_path = npy_file_path.replace(npy_basename, coords_npy_file_path)

        pos_labels = get_atoms_pos(pdb_file_path, aim_atom_types=["CA", "N", "C", "O"])
        _pos_labels = np.array(pos_labels[:])

        cube_data_array = np.load(npy_file_path)
        x1, x2, y1, y2, z1, z2, _ = np.load(coords_npy_file_path)
        upper_left_corner = [x1, y1, z1]
        lower_right_corner = [x2, y2, z2]

        # normalize coordinates
        cube_size = cube_data_array.shape
        try:
            for i, item_coord in enumerate(pos_labels):
                _, pos_labels[i] = self.normalize_coords(item_coord, _offset, upper_left_corner, lower_right_corner)
            relat_Ca_pos, relat_N_pos, relat_C_pos, relat_O_pos = pos_labels
        except ValueError:
            print(mrc_file_path, pdb_file_path)
            relat_Ca_pos, relat_N_pos, relat_C_pos, relat_O_pos = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
                                                                            [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], ])
            pos_labels = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], ])

        # print("After de-normalize: \t", de_normalize(pos_labels[0], upper_left_corner, lower_right_corner, _offset)            

        # rescale cube data
        cube_data_array = torch.from_numpy(self.rescale(cube_data_array))

        if self.gt_type == "heatmap":
            atoms_heatmaps = self.heatmap_generator.heatmap3D(cube_data_array, pos_labels)
            if self.mode == 1:
                return cube_data_array.unsqueeze(0),  atoms_heatmaps
            else:
                return cube_data_array.unsqueeze(0),  atoms_heatmaps, \
                        relat_Ca_pos, relat_N_pos, relat_C_pos, relat_O_pos, \
                       _pos_labels[0], _pos_labels[1], _pos_labels[2], _pos_labels[3], \
                        np.array(upper_left_corner), np.array(lower_right_corner), np.array(_offset)
        elif self.gt_type == "coords":
            if self.mode == 1:
                return cube_data_array.unsqueeze(0), relat_Ca_pos, relat_N_pos, relat_C_pos, relat_O_pos
            else:
                return cube_data_array.unsqueeze(0), relat_Ca_pos, relat_N_pos, relat_C_pos, relat_O_pos, \
                                                     _pos_labels[0], _pos_labels[1], _pos_labels[2], _pos_labels[3], \
                                                    np.array(upper_left_corner), np.array(lower_right_corner), np.array(_offset)
        else:
            print("The gt_type:{} is not defined".format(self.gt_type))
            raise ValueError

    def __len__(self):
        return len(self.mrc_cube_path)


class uniAminoTypeDataset(AminoAcidDataset):
    def __init__(self, amino_type, index_csv, standard_size=[32, 32, 32], zoom_type="diff"):
        super().__init__(index_csv, standard_size, zoom_type)
        self.aminoType = amino_type
        # update
        self.filter_amino_type(self.mrc_cube_path, self.pdb_cube_path)

    def filter_amino_type(self, mrc_file_path_list, pdb_file_path_list):
        filtered_mrc_paths = []
        filtered_pdb_paths = []
        offsets = []
        for i, item in enumerate(mrc_file_path_list):
            if self.aminoType in os.path.basename(item):
                filtered_mrc_paths.append(mrc_file_path_list[i])
                filtered_pdb_paths.append(pdb_file_path_list[i])
                offsets.append(self.offset_list[i])
        # update
        self.mrc_cube_path = filtered_mrc_paths
        self.pdb_cube_path = filtered_pdb_paths
        self.offset_list = offsets
    

    def normalize_coords(self, coords, offset, upper_left, lower_right, padding=2):
        return super().normalize_coords(coords, offset, upper_left, lower_right, padding)
    
    def set_mode(self, mode):
        return super().set_mode(mode)

    def rescale(self, cube_array):
        return super().rescale(cube_array)

    def __getitem__(self, item):
        return super().__getitem__(item)

    def __len__(self):
        return super().__len__()
    
    



# metrics
def my_criterial(l1_weight, l2_weight):
    return l1_weight * nn.L1Loss() + l2_weight * nn.MSELoss()



if __name__ == '__main__':
    # origin_size = [36, 38, 60]
    # aa = np.random.randint(0, 255, origin_size)
    # print(aa.shape)
    # print(aa[:10, :10, 1])
    
    # standard_size = [32, 32, 32]
    # scale = (standard_size[0] / origin_size[0],
    #          standard_size[1] / origin_size[1],
    #          standard_size[2] / origin_size[2],)
    # rest = zoom(aa, scale)
    # print("******************")
    # print(rest.shape)
    # print(rest[:10, :10, 1])

    EMdata_dir = '/mnt/data/zxy/amino-acid-detection/EMdata_dir/400_500/'
    mrc_fragment_dirs = '/mnt/data/zxy/amino-acid-detection/test_data/mrc_fragments/'
    pdb_fragment_dirs = '/mnt/data/zxy/amino-acid-detection/test_data/pdb_fragments/'
    generate_index_and_split_data(EMdata_dir, mrc_fragment_dirs, pdb_fragment_dirs, output_dir='./datas/')
    # split data into 3 "Trian, Valid, Test" datasets.
    split_data(pdb_index_csv='./datas/pdb_index.csv', mrc_index_csv='./datas/mrc_index.csv', output_dir='./datas/split/')
    

    # test heatmap generator
    # heatmap_generator = HeatmapGenerator3D(sigma=1.0)
    # print(heatmap_generator.kernel)
    # data_array = np.zeros([4, 4, 4])
    # print("***************************************")
    # pos = [[3, 3, 2.5]]
    # res = heatmap_generator.heatmap3D(data_array, pos)
    # print(res[0, 0, :, :])
    # print(res[0, 1, :, :])
    # print(res[0, 2, :, :])
    # print(res[0, 3, :, :])
    