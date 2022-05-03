"""
    Amino acid feature vector dataset.
    The dataset will use the model trained in "Amino-Pose-Estimation" part.

"""

from cProfile import label
from functools import total_ordering
from sklearn import datasets
from sklearn.utils import shuffle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# from cube_loader import AminoAcidDataset, de_normalize
# from atom_pose_estimation.inference import inference
# from layers import HourGlass3DNet
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import mrcfile 
import os
from scipy.ndimage.interpolation import zoom
# from utils import get_atoms_pos, parse_amino_acid
import random



AMINO_ACIDS = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
               'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']
AMINO_ACID_DICT = dict(zip(AMINO_ACIDS, range(20)))
print(AMINO_ACID_DICT)


def rescale(cube_array, standard_size=[16, 16, 16], zoom_type=None):
    """
    rescale a cube to a standard size, and return the rescaled coordinates
    """
    a, b, c = cube_array.shape
    # TODO: Check if the
    scale = (standard_size[0] / a,
             standard_size[1] / b,
             standard_size[2] / c,)
    if zoom_type =='same':
        # all dimensions zoomed at a same factor, so the output may not be the aim_size
        # need padding
        min_factor = min(scale)
        _rescaled = zoom(cube_array, min_factor)
        x, y, z = _rescaled.shape
        res = np.zeros([standard_size])
        res[(standard_size[0] - x) // 2:(standard_size[0] - x) // 2 + x,
            (standard_size[0] - y) // 2:(standard_size[0] - y) // 2 + y,
            (standard_size[0] - z) // 2:(standard_size[0] - z) // 2 + z] = _rescaled
        return res
    else:
        # all dimensions zoomed with different factors, so the output is the aim_size
        return zoom(cube_array, scale)


def de_normalize(related_pos, upper_left_corner, lower_right_corner, mrc_offset):
    """
    return a real pos of an atom
    :param cube_coords: [x1, x2, y1, y2, z1, z2]
    """
    # [x1, x2, y1, y2, z1, z2] = cube_coords
    # upper_left_corner = [x1, y1, z1]
    # lower_right_corner = [x2, y2, z2]
    cube_size = np.array(lower_right_corner) - np.array(upper_left_corner)
    # print("cube sizee: ", cube_size)
    # print(upper_left_corner)
    # print(related_pos)
    real_pos = np.array(related_pos) * cube_size + upper_left_corner
    x, y, z = real_pos
    # print(mrc_offset)
    _x = x/2 + mrc_offset[2]
    _y = y/2 + mrc_offset[1]
    _z = z/2 + mrc_offset[0]
    return np.array([_z, _y, _x])


def parse_offset(off_str):
    x, y, z = off_str.split(';')
    return float(x.strip()), float(y.strip()), float(z.strip())



# def generate_input_data(model, index_csv='../datas/split/train.csv', save_dir='/mnt/data/zxy/stage3_data/stage3-amino-keypoint-vectors/',
#                         absolute=True):
#     """
#         Args: Using the feature vector produced from Stage-2
#         TODO: Whether transoform the REAL coordinates to RELATIVE coordinates with respect to each protein.    ----> It's necessary.

#         Args:  
#             absulute: if True, after de-normalizing coords, add mrc_file_offset to it, to get the coordinates close to TRUE coord in PDB files.
#                       if False, no offset will be added 
#     """
#     model.eval()
#     # dataset = AminoAcidDataset(index_csv=index_csv, standard_size=[16, 16, 16],
#     #                            gt_type='coords')
#     # dataset.set_mode(2)
#     # data_loader = DataLoader(dataset)
#     df = pd.read_csv(index_csv)
#     for i in range(len(df)):
#         curr_records = df.iloc[i]
#         if absolute:
#             _offset = parse_offset(curr_records["offset"])
#         else:
#             _offset = [0.0, 0.0, 0.0] 
        
#         frag_id = curr_records['frag_id']
#         mrc_path = curr_records['mrc_path']
#         npy_file_path = mrc_path.replace('.mrc', '.npy')     
#         amino_type = os.path.basename(mrc_path)[:3]
#         amino_index = os.path.basename(mrc_path).strip('.mrc').split('_')[1]        # eg: */GLY_445.mrc

#         cube_data_array = np.load(npy_file_path)
#         cube_data_array = rescale(cube_data_array)
#         cube_data_tensor = torch.from_numpy(cube_data_array)
#         cube_data_tensor = cube_data_tensor.unsqueeze(0).unsqueeze(0)                # add dimension
#         # cube_data_tensor.to(device)

#         npy_basename = os.path.basename(npy_file_path)
#         coords_npy_file_path = npy_basename[:3] + '_coord' + npy_basename[3:]
#         coords_npy_file_path = npy_file_path.replace(npy_basename, coords_npy_file_path)
#         x1, x2, y1, y2, z1, z2, _ = np.load(coords_npy_file_path)
#         upper_left_corner = [x1, y1, z1]
#         lower_right_corner = [x2, y2, z2]

#         # Inference, and prepare feature vector for a SINGLE amino acid
#         Ca_output, N_output, C_output, O_output = model(cube_data_tensor)
#         pred_Ca = de_normalize(Ca_output[0].detach().cpu().numpy(), upper_left_corner, lower_right_corner, _offset)
#         pred_N  = de_normalize(N_output[0].detach().cpu().numpy(), upper_left_corner, lower_right_corner, _offset)
#         pred_C  = de_normalize(C_output[0].detach().cpu().numpy(), upper_left_corner, lower_right_corner, _offset)
#         pred_O  = de_normalize(O_output[0].detach().cpu().numpy(), upper_left_corner, lower_right_corner, _offset)
#         # integrate type and 4 key point atoms positon
#         fea_vec = np.zeros(32)      # 20  +  4 * 3
#         fea_vec[-12:] = np.concatenate([pred_Ca, pred_N, pred_C, pred_O])
#         fea_vec[AMINO_ACID_DICT[amino_type]] = 1.0

#         # save, 
#         _save_dir = os.path.join(save_dir, frag_id[:4])
#         if not os.path.exists(_save_dir):
#             os.makedirs(_save_dir)
#         _save_path = os.path.join(_save_dir, os.path.basename(npy_file_path))
#         print(_save_path)
#         np.save(_save_path, fea_vec)


# def generate_input_simulation_data(pdb_index_csv):
#     pass


# def get_amino_type_from_pdb(pdb_file_path):
#     with open(pdb_file_path, 'r') as pdb_file:
#         for line in pdb_file:
#             if line.startswith("ATOM"):
#                 amino_acid = parse_amino_acid(line)
#                 break
#     return amino_acid


# def generate_output_data(src_dir="/mnt/data/zxy/amino-acid-detection/test_data/pdb_fragments/", 
#                          dst_dir='/mnt/data/zxy/stage3_data/stage3_labels/'):
#     """
#     Generate an amino-sequence (label) data for each protein;
#     The data has following format:
#         [[Type-One-Hot-Code, Ca-Pos, N-Pos, C-Pos, O-pos], ...]
#     Each row represents a single amino-acid, and the order will be arranged by Proteins' amino-acid-sequence

#     Args: 
#         src_dir: directory which save protein datas, each sub-directory in this is named with PROTEIN_ID
#         dst_dir: directory which save the OUTPUT data 

#     IF: Append Amino-Type-Confidence
#     """
#     protein_ids = sorted(os.listdir(src_dir))
#     for p_id in protein_ids:
#         output_dir = os.path.join(dst_dir, p_id)
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
#         print(output_dir)
#         # prepare data for a protein
#         curr_dir = os.path.join(src_dir, p_id)
#         pdb_files = os.listdir(curr_dir)
#         index_list = []
#         array_list = []
#         for pdb_name in os.listdir(curr_dir):
#             index_list.append(int(pdb_name.split('.')[0]))
#             amino_type= get_amino_type_from_pdb(os.path.join(curr_dir, pdb_name))
#             ca, n, c, o = get_atoms_pos(os.path.join(curr_dir, pdb_name))

#             feature_vec = [0.0] * 32
#             feature_vec[AMINO_ACID_DICT[amino_type]] = 1.0
#             feature_vec[20:23] = ca
#             feature_vec[23:26] = n
#             feature_vec[26:29] = c
#             feature_vec[29:32] = o
#             array_list.append(feature_vec)
#         order = np.argsort(np.array(index_list))
#         ordred_fea_vecs = np.array(array_list)[order]
#         # print(ordred_fea_vecs[:5])
#         np.save(os.path.join(output_dir, "{}.npy".format(p_id)), ordred_fea_vecs)


def generate_data_index(fea_src_dir='/mnt/data/zxy/stage3_data/stage3-amino-keypoint-vectors/', 
                        label_src_dir='/mnt/data/zxy/stage3_data/stage3_labels/',
                        val_split=0.3, output_dir='../datas/tracing_data'
                        ):
        pdb_id_list = os.listdir(fea_src_dir)
        df = {"p_id": pdb_id_list}
        acid_num_list = []
        for p_id in pdb_id_list:
            acid_num_list.append(len(os.listdir(os.path.join(fea_src_dir, p_id))))
        df['acid_num'] = acid_num_list
        df = pd.DataFrame(df)
        def add_prefix1(str_):
            return fea_src_dir + str_
        def add_prefix2(str_):
            return label_src_dir + str_
        # df = df['featurePath'].map(add_prefix1)
        # df = df['label_path'].map(add_prefix2)
        print(df.head(6))
        print("Data length:", len(df))
        print("MAX LENGTH: ", max(df['acid_num']))


        train_idx, val_idx = train_test_split(list(range(len(df))), test_size=val_split)
        _train_df = df.iloc[train_idx].reset_index(drop=True)
        valid_df = df.loc[val_idx].reset_index(drop=True)

        train_idx, test_idx = train_test_split(list(range(len(_train_df))), test_size=0.3)
        train_df = _train_df.iloc[train_idx].reset_index(drop=True)
        test_df = _train_df.iloc[test_idx].reset_index(drop=True)

        # save
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        train_df.to_csv(os.path.join(output_dir, "train.csv"), index=None)
        valid_df.to_csv(os.path.join(output_dir, "valid.csv"), index=None)
        test_df.to_csv(os.path.join(output_dir, "test.csv"), index=None)

        print("Train sets len:{} \t Valid sets len: {}\t Test sets len: {}".format(len(train_df), len(valid_df),
                                                                                   len(test_df)))


def random_shift_and_rotate(data_array, radius=0.1, shift=[0.1, 0.1, 0.1], theta=1.0, gamma=1.0):
    """
    A data argumentation method, 
        data_array: 1 x 13D vector
        radius: rotation radius
        theta: rotatation angle_1
        gamma: rotatation angle_2
    """
    pass



class AminoFeatureDataset(Dataset):
    def __init__(self, index_csv, 
                 fea_src_dir='/mnt/data/zxy/relat_coors_stage3-amino-keypoint-vectors/', 
                 label_src_dir='/mnt/data/zxy/stage3_data/stage3_labels/',
                 params_file='../datas/params/coord_mean_std.json',
                 max_len = 512, dims = 13, 
                 input_shuffle=True, z_score_coords=False,
                 pad_idx=0,
                 padding=True, padding_form=None, zero_center = False,
                 zoom_type="diff"):
        self.feature_src_dir = fea_src_dir
        self.labels_src_dir = label_src_dir
        self.max_len = max_len
        self.dims = dims
        self.shuffle = input_shuffle
        self.zero_center = zero_center
        self.padding_form = padding_form
        self.z_score_coords = z_score_coords
        if z_score_coords:
            self.relative_coords_mean, self.relative_coords_std = [0.2861679701546054, 0.05801937740728386]
            #   0.5723359403092108, 0.11603875481456773
        self.index_csv = index_csv
        self.pid = pd.read_csv(index_csv)["p_id"]
        self.pid_visable = False
    
    def set_visiable(self, states=False):
        self.pid_visable = states
    
    def __getitem__(self, index: int):
        """
            Add BOS, EOS and padding for each protein(labels token).
        """
        pid = self.pid[index]
        if self.pid_visable:
            print("curr protein id is : {}".format(pid))
            
        protein_fea_dir = os.path.join(self.feature_src_dir, pid)
        protein_label_dir = os.path.join(self.labels_src_dir, pid)
        
        amino_file_list = os.listdir(protein_fea_dir)
        detected_amino_nums = len(amino_file_list)
        if self.shuffle:
            random.shuffle(amino_file_list)
        
        data_array = np.zeros([self.max_len, self.dims])
        # print("Data array: shape: ", data_array.shape)
        for i, amino_file in enumerate(amino_file_list):
            # temp = np.load(os.path.join(protein_fea_dir, amino_file)).reshape(-1)
            # print(temp.shape)
            data_array[i] = np.load(os.path.join(protein_fea_dir, amino_file)).reshape(-1)
        
        if self.z_score_coords:
            data_array[:detected_amino_nums][1:] = (data_array[:detected_amino_nums][1:] - self.relative_coords_mean) / self.relative_coords_std
        
        # # add BOS
        # data_array[1:] = data_array[:-1]
        # data_array[0] = np.array([22.0] + [0.0] * 12)

        # # add EOS
        # data_array[detected_amino_nums][0] = 21.0
        # add padding

        if self.padding_form is not None:
            for j in range(detected_amino_nums+1, self.max_len):
                data_array[j] = self.padding_form
        
        # prepare output / Decoder Input
        label_file = os.path.join(protein_label_dir, "{}.npy".format(pid))
        label_vec = np.load(label_file)
        label_amino_nums = len(label_vec)

        # normalize coordinates
        if self.z_score_coords:
            label_vec[:, 1:] = (label_vec[:, 1:] - self.relative_coords_mean) / self.relative_coords_std
        
        if self.zero_center:
            label_vec[:, 1:] = (label_vec[:, 1:] - 0.5) / 0.5

        label_array = np.zeros([self.max_len, self.dims])
        label_array[1:label_amino_nums+1] = label_vec                          

        # # add BOS
        # label_array[1:] = data_array[:-1]
        label_array[0][0] = 22.0
        # add EOS
        label_array[label_amino_nums][0] = 21.0   
        
        return data_array, label_array

    def __len__(self) -> int:
        # print("{}:\t{}".format(self.index_csv, len(self.pid)))
        return len(self.pid)


class AminoLinkageDataset(Dataset):
    def __init__(self, index_csv, 
                 fea_src_dir='/mnt/data/zxy/relat_coors_stage3-amino-keypoint-vectors/', 
                 label_src_dir='/mnt/data/zxy/stage3_data/stage3_labels/',
                 params_file='../datas/params/coord_mean_std.json',
                 max_len = 512, dims = 13, 
                 input_shuffle=True, z_score_coords=False,
                 pad_idx=0,
                 padding=True, padding_form=None, zero_center = False,
                 zoom_type="diff"):
        self.feature_src_dir = fea_src_dir
        self.labels_src_dir = label_src_dir
        self.max_len = max_len
        self.dims = dims
        self.shuffle = input_shuffle
        self.zero_center = zero_center
        self.padding_form = padding_form
        self.z_score_coords = z_score_coords
        if z_score_coords:
            self.relative_coords_mean, self.relative_coords_std = [0.2861679701546054, 0.05801937740728386]
            #   0.5723359403092108, 0.11603875481456773
        self.index_csv = index_csv
        self.pid = pd.read_csv(index_csv)["p_id"]
        self.pid_visable = False
    
    def set_visiable(self, states=False):
        self.pid_visable = states
    
    def __getitem__(self, index: int):
        """
            Add BOS, EOS and padding for each protein(labels token).
        """
        pid = self.pid[index]
        if self.pid_visable:
            print("curr protein id is : {}".format(pid))
            
        protein_fea_dir = os.path.join(self.feature_src_dir, pid)
        protein_label_dir = os.path.join(self.labels_src_dir, pid)
        
        amino_file_list = os.listdir(protein_fea_dir)
        detected_amino_nums = len(amino_file_list)
        if self.shuffle:
            random.shuffle(amino_file_list)
        
        data_array = np.zeros([self.max_len, self.dims])
        # print("Data array: shape: ", data_array.shape)
        for i, amino_file in enumerate(amino_file_list):
            # temp = np.load(os.path.join(protein_fea_dir, amino_file)).reshape(-1)
            # print(temp.shape)
            data_array[i] = np.load(os.path.join(protein_fea_dir, amino_file)).reshape(-1)
        
        if self.z_score_coords:
            data_array[:detected_amino_nums][1:] = (data_array[:detected_amino_nums][1:] - self.relative_coords_mean) / self.relative_coords_std

        if self.padding_form is not None:
            for j in range(detected_amino_nums+1, self.max_len):
                data_array[j] = self.padding_form
        
        # prepare output / Decoder Input
        label_file = os.path.join(protein_label_dir, "{}.npy".format(pid))
        label_vec = np.load(label_file)
        label_amino_nums = len(label_vec)

        # normalize coordinates
        if self.z_score_coords:
            label_vec[:, 1:] = (label_vec[:, 1:] - self.relative_coords_mean) / self.relative_coords_std
        
        if self.zero_center:
            label_vec[:, 1:] = (label_vec[:, 1:] - 0.5) / 0.5

        label_array = np.zeros([self.max_len, self.dims])
        label_array[:label_amino_nums] = label_vec                          

        
        return data_array, label_array

    def __len__(self) -> int:
        # print("{}:\t{}".format(self.index_csv, len(self.pid)))
        return len(self.pid)



class UniProtein():
    def __init__(self, protein_id, max_len=512,
                block_index=0,
                pad_index=2,
                fea_src_dir='/mnt/data/zxy/relat_coors_stage3-amino-keypoint-vectors/', 
                label_src_dir='/mnt/data/zxy/stage3_data/stage3_labels/',
                shuffle=False, dist_rescaling=True) -> None:
        
        self.protein_id = protein_id
        self.data_dir = os.path.join(fea_src_dir, protein_id)
        self.label_file = os.path.join(label_src_dir, protein_id, "{}.npy".format(protein_id))
        self.label_index_file = os.path.join(label_src_dir, protein_id, "{}_amino_seq_idxs.npy".format(protein_id))

        self.tracked_acid_num = len(os.listdir(self.data_dir))
        # self.gt_acid_num = amino_acids_num
        self.max_len = max_len
        self.shuffle = shuffle
        self.dist_rescaling = dist_rescaling
        self.linkage_gt_square = None
        self.linkage_gt_sets   = None
        self.rand_index = None
        self.rand_index_vec = None
        self.shuffled_square_gt = None
        self.detected_index_vec = []

        self.pad_idx = pad_index
        
        self.load_tracked_amino_acids()
        self.load_gt_amino_acids()
        if shuffle and len(self.data_array) != len(self.label_data):
            """
            [6QP4, 6TPW, 6PGW, 7B1E, 7B1P, 7B1Q, ]
            """
            print("Protein-{}'s data_array is replaced by ground_truth_data".format(self.protein_id))
            self.data_array = self.label_data

        self.load_gt_amino_seq_index()
        self.construct_square_gt()

        if shuffle:
            self.shuffle_square_gt()
    
    def load_tracked_amino_acids(self, padding=False, pad=None):
        if pad == None:
            pad = -1
        
        if padding:
            data_array = np.zeros((self.max_len, 13)) + pad
        else:
            data_array = np.zeros((self.tracked_acid_num, 13))
        
        amino_file_list = os.listdir(self.data_dir)
        self.amino_file_list = np.array(amino_file_list)

        # TODO: When sorted, store the sorted index
        # if self.shuffle:
        #     random.shuffle(amino_file_list)

        # relative coodinates
        for i, amino_file in enumerate(amino_file_list):
            data_array[i] = np.load(os.path.join(self.data_dir, amino_file)).reshape(-1)
        
        self.data_array = data_array
        self.amino_types_list = data_array[:, 0]

    def load_gt_amino_acids(self, padding=False, pad=None):
        """
        NOTE: detected amino-acid nums may not equal to the ground truth amino-acid nums
              1. Some amino-acids are not detected 
              2. Some amino-acid are not located by the .pdb files(from experiments) 
        """
        label_vec = np.load(self.label_file)

        self.gt_amino_num = len(label_vec)
        # print("Protein: {}\t Tracked Nums: {}\t GT Nums: {}".format(
        #     self.protein_id, self.tracked_acid_num, self.gt_amino_num))
        self.label_data = label_vec

    def load_gt_amino_seq_index(self, ):
        index_vec = np.load(self.label_index_file)
        index_vec = np.sort(index_vec)

        # for some indices is less than 0 
        # after sorting, the index_vec[0] is the minimum
        if index_vec[0] < 0:
            index_vec = index_vec - index_vec[0] + 1

        exists = np.zeros(index_vec[-1])
        exists[index_vec - 1] = 1
        # try:
        #     exists[index_vec - 1] = 1
        # except IndexError:
        #     print("In protein {}, the min amino: idx {} \t max amino idx: {}".format(
        #         self.protein_id, index_vec[0], index_vec[-1]))
        self.index_vec = index_vec
        missing_idxs = np.where(exists == 0)[0]
        
        self.located_amino_index = index_vec
        self.missing_amino_index = missing_idxs
        self.existence_array = exists

    
    def construct_square_gt(self):
        # index from 1, to the END
        # The end is assumed to be the located amino

        # 1 means linkage, while '0' means no linkage
        # 2 as the padding index

        # TODO: ATTENTION
        # The length is equal to the detected animo-acid nums(corresponding to GT)
        # But not equal to the real animo-acid nums with a protein, because many missed
        # amino-acids are not located by experiments
        length = len(self.index_vec)

        gt = np.zeros((length, length))

        for i in range(1, length):
            gt[i, i-1] = 1
            gt[i-1, i] = 1

        for i in range(1, length):
            item_index = self.index_vec[i]
            item_existence = self.existence_array[item_index - 1]
            if item_existence == 0:
                gt[i, i - 1] = 0
                gt[i - 1, i] = 0

        self.linkage_gt_square = gt
    
    def shuffle_square_gt(self):
        rand_idx = np.arange(len(self.data_array))                # 
        # assert len(self.data_array) == len(self.index_vec)          # May not true, some not tracked
        # if len(self.data_array) != len(self.index_vec):
        #     print("{}: data array length is not equal to index_vec length".format(self.protein_id))
        #     raise ValueError
        # rand_idx = np.arange(len(self.index_vec))
        np.random.shuffle(rand_idx)

        self.rand_idx = rand_idx
        self.rand_index_vec = self.index_vec[rand_idx]              # index 375 is out of bounds for axis 0 with size 375

        _gt = self.linkage_gt_square[rand_idx].T
        _gt = _gt[rand_idx].T

        self.shuffled_square_gt = _gt


    def construct_linkage_set(self):
        linkage_list = []
        length = self.index_vec[-1]

        if self.linkage_gt_square == None:
            self.construct_linkage_set()

        for i in range(1, length):
            if self.linkage_gt_square[i, i-1]:
                linkage_list.append([i, i-1])
        
        self.inkage_gt_sets = linkage_list


class LinkageSet(Dataset):
    def __init__(self, index_csv, 
                max_len=512, pad_idx=2, 
                fea_src_dir='/mnt/data/zxy/relat_coors_stage3-amino-keypoint-vectors/', 
                label_src_dir='/mnt/data/zxy/stage3_data/stage3_labels/',
                using_gt = False,
                shuffle=False, dist_rescaling=True) -> None:
        super().__init__()

        self.index_csv = index_csv
        self.pids = pd.read_csv(index_csv)["p_id"]
        self.pid_visable = False
        
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.using_gt = using_gt
        self.shuffle = shuffle

    def set_visiable(self, states=False):
        self.pid_visable = states
    
    def add_padding(self, data, dims=1):
        _len, dim = data.shape
        assert _len < self.max_len
        if dims == 1:
            result = np.zeros((self.max_len, dim)) + self.pad_idx
            result[:_len] = data
            return result
        elif dims == 2:
            result = np.zeros((self.max_len, self.max_len)) + self.pad_idx
            result[:_len, :dim] = data
            return result
    
    def __getitem__(self, index):
        t_protein = UniProtein(self.pids[index], pad_index=self.pad_idx, shuffle=self.shuffle)
        index_vec = t_protein.index_vec
        # print("All amino nums in Protein {} is {}".format(self.pids[index], index_vec[-1]))

        amino_data_array = t_protein.data_array             # estimated data, extract from detected cube
        amino_data_gt = t_protein.label_data                # ground truth data, loaded from the .pdb file
        linkage_gt = t_protein.linkage_gt_square
        
        amino_nums_det = len(amino_data_array)
        amino_nums_gt = len(amino_data_gt)

        assert len(amino_data_array) <= self.max_len
        assert len(amino_data_gt) <= self.max_len

        # add shuffling
        if self.shuffle:
            # det_index = np.arange(amino_nums_det)
            # gt_index = np.arange(amino_nums_gt)
            # np.random.shuffle(det_index)
            # np.random.shuffle(gt_index)
            shuffled_index_vec = t_protein.rand_index_vec
            shuffled_index = t_protein.rand_idx
            amino_data_array = amino_data_array[shuffled_index]
            linkage_gt = t_protein.shuffled_square_gt

        # also can shuffle after padding
        _amino_data_array = self.add_padding(amino_data_array)
        _amino_data_gt = self.add_padding(amino_data_gt)
        _linkage_gt = self.add_padding(linkage_gt, dims=2)

        if self.using_gt:
            return _amino_data_gt, _linkage_gt, amino_nums_gt

        return _amino_data_array, _linkage_gt, amino_nums_det


    def __len__(self) -> int:
        return len(self.pids)


if __name__ =='__main__':
    gpu_id = "0, 1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print('gpu ID is ', str(gpu_id))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model = torch.load('./checkpoints/Hourglass3D_Regression/2022-03-07observation_11.00.06/best_HG3_CNN.pt', map_location='cpu')
    # model.to(device)
    # generate_input_data(model, index_csv='../datas/split/train.csv', save_dir='/mnt/data/zxy/relat_coors_stage3-amino-keypoint-vectors/')
    # generate_input_data(model, index_csv='../datas/split/test.csv', save_dir='/mnt/data/zxy/relat_coors_stage3-amino-keypoint-vectors/')
    # generate_input_data(model, index_csv='../datas/split/valid.csv', save_dir='/mnt/data/zxy/relat_coors_stage3-amino-keypoint-vectors/')
    # print(model.state_dict())
    
    # # test generated data
    # fea_npy_dir = "/mnt/data/zxy/stage3-amino-keypoint-vectors/6OD0/"
    # file_list = os.listdir(fea_npy_dir)
    # for _fn in file_list:
    #     data = np.load(os.path.join(fea_npy_dir, _fn))
    #     print("{} -- {}".format(_fn, data[20:]))
    # data = np.load(fea_nmpy_path)
    # print(data)
    # print('--------------------')
    # print(data[:20])
    # print(data[20:23], data[23:26], data[26:29], data[29:32])

    # # generate_output_data()
    # generate_data_index(output_dir='../datas/tracing_data2')


    # the_dataset = AminoFeatureDataset(index_csv='../datas/tracing_data/test.csv')
    # the_loader  = DataLoader(the_dataset, batch_size=2)
    # for idx, data in enumerate(the_loader):
    #     seq_data_array = data[0].to(torch.float32).to(device)
    #     labels = data[1].to(torch.float32).to(device)
    #     print(seq_data_array.shape)
    #     print(labels.shape)
    #     print("***********************\n")
    #     print(seq_data_array[0])
    #     print(seq_data_array[1])
    #     print(labels[0])
    #     print(labels[1])


    #     s1 = seq_data_array[0]
    #     s2 = seq_data_array[1]
    #     print("\nSUM-1: {}".format((s1 == s2).sum()))
    #     l1 = labels[0]
    #     l2 = labels[1]
    #     print("\nSUM-2: {}".format((l1 == l2).sum()))
    #     break



    # Test shuffle
    # protein = UniProtein("6PGQ", shuffle=True)
    protein = UniProtein("6PGW", shuffle=True)
    print(protein.gt_amino_num)
    print("*******************")

    print(protein.index_vec)
    print("*******************")

    print(protein.rand_index_vec)
    print("*******************")
    print(protein.linkage_gt_square)
    print("*******************")
    print(protein.shuffled_square_gt)

    data_array = protein.data_array
    data_arr_gt = protein.label_data
    origin_gt = protein.linkage_gt_square
    shuffle_gt = protein.shuffled_square_gt
    origin_index_vec = protein.index_vec
    existence_vec = protein.existence_array
    rand_index_vec = protein.rand_index_vec
    rand_index = protein.rand_idx

    print("detected nums: {} \t gt nums: {} \t index_vec_len: {} \t rand_idx len: {} \t existence_len: {} \t sum_of_existence: {}".format(
        len(data_array), len(data_arr_gt), len(origin_index_vec), len(rand_index), len(existence_vec), np.sum(existence_vec)))

    print("Origin sum: {} \t Shuffled Sum: {}".format(np.sum(origin_gt), np.sum(shuffle_gt)))

    # map_dict = dict(zip(protein.rand_index, protein.index_vec))
    print(rand_index)
    map_dict = {}
    for i in range(len(protein.index_vec)):
         temp = np.where(rand_index == i)[0]
         map_dict[i] = temp[0]
    print(map_dict)

    for i in range(0, len(protein.index_vec) - 1):
        j = i + 1
        print("Ori: {} \t Shuffled: {}".format(origin_gt[i, j], shuffle_gt[map_dict[i], map_dict[j]]))



