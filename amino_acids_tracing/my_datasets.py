"""
    Reconstruct Linkage-Dataset after endless breakdown
"""

from matplotlib.pyplot import axis
from sklearn import datasets
from sklearn.linear_model import PassiveAggressiveClassifier
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
import os, json
from scipy.ndimage.interpolation import zoom
# from utils import get_atoms_pos, parse_amino_acid
import random

AMINO_ACIDS = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
               'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']
AMINO_ACID_DICT = dict(zip(AMINO_ACIDS, range(20)))


class UniProtein():
    """
    1. load tracked amino data:
        if not re_generate: load from a single protein_id.npy
    2. load ground truth amino-data
        if not re_generate: load from a single protein_id.npy
    3. generate linkage state graph(affinity matrix) ---- 3 classes:
        "pre-stored if use"
        a. 0 means no linkage
        b. 1 means former linkage.
        c. 2 means latter linkage
        eg: [A, B, C], the linkage matrix is :
                [[0, 1, 0]
                [2, 0, 1]
                [0, 2, 1]]
        which means: A is pre-linked to B, and B is sub-linked to A
                     B is pre-linked to C, and C is sub-linked to B
    3.1 generate linkage state graph(affinity matrix) ---- 2 classes
    4. partly choose aminos and there linkage states
    5. shuffle the amino-acid samples, and correspondingly, their linkage ground truth
    6. store amino-index vectors
    7. Do coordinates normalization
    """
    def __init__(self, protein_id, max_len=512,
                block_index=0,
                re_generate=False, 
                fea_src_dir='/mnt/data/zxy/relat_coors_stage3-amino-keypoint-vectors/', 
                label_src_dir='/mnt/data/zxy/stage3_data/stage3_labels/',
                label_linkage_dir='/mnt/data/zxy/stage3_data/stage3_labels/',
                linakge_shuffle=False, dist_rescaling=True,
                sample_num = None,
                random_crop=False, crop_bins=10) -> None:
        
        self.protein_id = protein_id
        self.data_dir = os.path.join(fea_src_dir, protein_id)
        self.label_file = os.path.join(label_src_dir, protein_id, "{}.npy".format(protein_id))
        self.label_index_file = os.path.join(label_src_dir, protein_id, "{}_amino_seq_idxs.npy".format(protein_id))
        self.label_linkage_file = os.path.join(label_linkage_dir, protein_id, "{}_linkage.npy".format(protein_id))
        self.detected_data_array_file = os.path.join(label_src_dir, protein_id, "{}_detected_data_array.npy".format(protein_id))
        self.detected_data_index_file = os.path.join(label_src_dir, protein_id, "{}_detected_data_index.npy".format(protein_id))
        self.detected_data_linkaga_file = os.path.join(label_src_dir, protein_id, "{}_detected_data_linkage.npy".format(protein_id))

        self.detected_acid_num = len(os.listdir(self.data_dir))
        # self.gt_acid_num = amino_acids_num
        self.max_len = max_len
        self.block_index = block_index                      # if dropout, the index of dropouted block
        self.re_generate = re_generate                      # if re-generate, generate data_array, linkage_gt, and store them from initial data
        self.shuffle = linakge_shuffle                      # whether shuffle the data
        self.sample_num = sample_num                        # if sample the num must larger than 1, and sampleing after shuffling; if None, no sampled used 
        self.linkage_gt_square = None                       # linkage ground truth, "affinity matrix" format
        self.linkage_gt_sets   = None                       # linkage ground truth, "linkage sets"  format
        self.rand_index = None                              # if shuffle, the shuffled index of data array and 
        self.rand_index_vec = None                          # after shuffling, the index of each rows in self.data_array(detected or ground truth data_array)
        self.shuffled_square_gt = None
        self.detected_index_vec = None
        self.linkage_detected_square = None                 # linakge(affinity matrix) constucted from detected detected amino-acids
        self.shuffled_data_array = None
        self.shuffled_label_data = None

        # data argumentation
        self.random_crop = random_crop
        self.crop_bins = crop_bins
        
        self.load_tracked_amino_acids()                     # Make sure please: the data_array are sorted by their amino_index
        self.load_gt_amino_acids()                          # The gound truth amino acid num may not equal to the detectd num
        if self.shuffle and len(self.data_array) != len(self.label_data):
            """
            [6QP4, 6TPW, 6PGW, 7B1E, 7B1P, 7B1Q, ]  ---> There are some bugs in the these protein data
            """
            print("Protein-{}'s data_array is replaced by ground_truth_data".format(self.protein_id))
            self.data_array = self.label_data

        self.load_gt_amino_seq_index()
        self.construct_ground_truth_square_gt()
        self.construct_detected_data_square_gt()

        if self.random_crop:
            i, block_len= self.linkage_rand_crop()
            self.linkage_gt_square[:, i:i+block_len] = 0            # TODO: DY- Set to 2? 
            self.linkage_gt_square[i:i+block_index] = 0
            self.data_array[i:i+block_len] = 0

        if self.shuffle:
            self.shuffle_square_gt()
    
    def load_tracked_amino_acids(self):
        if not self.re_generate:
            data_array = np.load(self.detected_data_array_file)
            self.data_array = data_array
            amino_index_list = np.load(self.detected_data_index_file)
            self.detected_index_vec = amino_index_list
            return None

        amino_file_list = os.listdir(self.data_dir)
        self.amino_file_list = np.array(amino_file_list)                    # sort ?  In context with PDB order ? 
        # relative coodinates
        data_array = np.zeros((self.detected_acid_num, 13))
        amino_index_list = []
        for i, amino_file in enumerate(amino_file_list):
            data_array[i] = np.load(os.path.join(self.data_dir, amino_file)).reshape(-1)
            amino_index_list.append(int(amino_file[4:].split('.')[0]))
        amino_index_list = np.array(amino_index_list)
        self.detected_index_vec = amino_index_list

        order = np.argsort(amino_index_list)
        # NOTE: It must be sorted if shuffle the data                       # DY: ATTENTION
        data_array = data_array[order]
        _amino_index_list = amino_index_list[order]
        np.save(self.detected_data_array_file, data_array)
        np.save(self.detected_data_index_file, _amino_index_list)
        print("Finished: {}, {}".format(self.detected_data_array_file, self.detected_data_index_file))

        self.data_array = data_array
        self.amino_types_list = data_array[:, 0]

    def load_gt_amino_acids(self):
        """
        NOTE: detected amino-acid nums may not equal to the ground truth amino-acid nums
              1. Some amino-acids are not detected 
              2. Some amino-acid are not located by the .pdb files(from experiments) 
        """
        label_vec = np.load(self.label_file)

        self.gt_amino_num = len(label_vec)
        self.label_data = label_vec

    def load_gt_amino_seq_index(self, ):
        index_vec = np.load(self.label_index_file)                  # The index may be negative value like -1, -2
        # ATTENTION:  This is essential to the whole process
        index_vec = np.sort(index_vec)                              # DY: Is sorted by the order like [1, 2, 3, 4, 5...] ?

        # for some indices is less than 0 
        # after sorting, the index_vec[0] is the minimum
        if index_vec[0] < 0:
            index_vec = index_vec - index_vec[0] + 1                # Make the index is from 1

        exists = np.zeros(index_vec[-1])
        exists[index_vec - 1] = 1

        self.index_vec = index_vec
        missing_idxs = np.where(exists == 0)[0]
        
        self.located_amino_index = index_vec
        self.missing_amino_index = missing_idxs
        self.existence_array = exists

    def construct_ground_truth_square_gt(self):
        """
        TODO: the linkage_gt is constructed from Ground-Trurh-Label-Data
              but not from the detcted animo-acids
        """
        if not self.re_generate:
            self.linkage_gt_square = np.load(self.label_linkage_file)
            return None
            
        length = len(self.index_vec)
        gt = np.zeros((length, length))

        for i in range(1, length):
            index_diff = self.index_vec[i] - self.index_vec[i-1]
            if index_diff == 1:
                gt[i, i-1] = 1      # former linkage  
                gt[i-1, i] = 2      # latter linakge
            else:
                pass
        np.save(self.label_linkage_file, gt)
        print("Finished: {}".format(self.label_linkage_file))
        self.linkage_gt_square = gt

    def construct_detected_data_square_gt(self):
        if not self.re_generate:
            self.linkage_detected_square = np.load(self.detected_data_linkaga_file)
            return None

        length = len(self.detected_index_vec)
        gt = np.zeros((length, length))

        for i in range(1, length):
            index_diff = self.detected_index_vec[i] - self.detected_index_vec[i-1]
            if index_diff == 1:
                gt[i, i-1] = 1      # former linkage  
                gt[i-1, i] = 2      # latter linakge
            else:
                pass
        np.save(self.detected_data_linkaga_file, gt)
        print("Finished: {}".format(self.detected_data_linkaga_file))
        self.linkage_detected_square = gt    

    def shuffle_square_gt(self):
        rand_idx = np.arange(len(self.data_array))                # 
        np.random.shuffle(rand_idx)
        self.rand_idx = rand_idx
        self.rand_index_vec = self.index_vec[rand_idx]              # index 375 is out of bounds for axis 0 with size 375

        _gt = self.linkage_gt_square[rand_idx][:, rand_idx]
        # _gt = self.linkage_gt_square[rand_idx].T
        # _gt = _gt[rand_idx].T

        self.shuffled_data_array = self.data_array[rand_idx]                 # Attentention, No Need to Shuffle TWICE !!!
        self.shuffled_label_data = self.label_data[rand_idx]
        self.shuffled_square_gt = _gt
    
    def linkage_rand_crop(self):
        bins = self.crop_bins
        linkage_length = len(self.data_array)

        block_len = linkage_length // bins
        # randomly choose part
        i = random.randint(0, linkage_length - block_len)

        return i, block_len

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
    """
    For each extracted amino-acid feature
    1. Load data_array or shuffled data_array
    2. Load linkage_ground_truth or shuffled linkage_gt
    3. Set if use multi-class classification
    4. add padding 
    5. Coordinates Normalization
    """
    def __init__(self, index_csv, 
                max_len=512, pad_idx=-1, 
                fea_src_dir='/mnt/data/zxy/relat_coors_stage3-amino-keypoint-vectors/', 
                label_src_dir='/mnt/data/zxy/stage3_data/stage3_labels/',
                using_gt = False, multi_class=False,
                shuffle=False, dist_rescaling=True,
                coors_norm=False, 
                random_crop=False, crop_bins=10) -> None:
        super().__init__()

        self.index_csv = index_csv
        self.pids = pd.read_csv(index_csv)["p_id"]
        self.pid_visable = False
        
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.using_gt = using_gt

        self.random_crop = random_crop
        self.crop_bins = crop_bins
        self.multi_class = multi_class
        self.shuffle = shuffle
        self.coords_norm = coors_norm


    def set_visiable(self, states=False):
        self.pid_visable = states
    
    def add_padding(self, data, _pad_idx=None, dims=1):
        _len, dim = data.shape

        if _pad_idx == None:
            _pad_idx = self.pad_idx

        assert _len < self.max_len
        if dims == 1:
            result = np.zeros((self.max_len, dim)) + _pad_idx
            result[:_len] = data
            return result
        elif dims == 2:
            result = np.zeros((self.max_len, self.max_len)) + _pad_idx
            result[:_len, :dim] = data
            return result
    
    def coordinates_normalization(self, datas, mean_std_params_file='../datas/params/coord_mean_std.json'):
        length, dims = datas.shape
        assert dims == 13
        with open(mean_std_params_file, 'r')  as j:
            _params = json.load(j)
        
        x_mean_std = _params['x_mean_std']                  # TODO: Of cource, there face the same problem, the "X" axis and "Z" axis may be disordered
        y_mean_std = _params['y_mean_std']                  # But the mean and std are not far from in X and Z, so, the impact is limmited in some way
        z_mean_std = _params['z_mean_std']                  # DY: ATTENTION
        
        datas[:, 1::3] = (datas[:, 1::3] - x_mean_std[0])  / x_mean_std[1]
        datas[:, 2::3] = (datas[:, 2::3] - y_mean_std[0])  / y_mean_std[1]
        datas[:, 3::3] = (datas[:, 3::3] - z_mean_std[0])  / z_mean_std[1]

        return datas

    def __getitem__(self, index):
        t_protein = UniProtein(self.pids[index], 
                               re_generate=False,
                               linakge_shuffle=self.shuffle,
                               random_crop=self.random_crop, crop_bins=self.crop_bins)
        index_vec = t_protein.index_vec
        if self.pid_visable:
            print("Current Protein ID: ", self.pids[index])
        # print("All amino nums in Protein {} is {}".format(self.pids[index], index_vec[-1]))
        if not self.shuffle:
            amino_data_array = t_protein.data_array             # estimated data, extract from detected cube; Maybe shuffled if the 't_protein' has the 'shuffle' params 
            amino_data_gt = t_protein.label_data                # ground truth data, loaded from the .pdb file
            linkage_gt = t_protein.linkage_gt_square
            self.index_vec = t_protein.index_vec
        else:
            amino_data_array = t_protein.shuffled_data_array
            amino_data_gt = t_protein.shuffled_label_data
            linkage_gt = t_protein.shuffled_square_gt
            self.index_vec = t_protein.rand_index_vec
        
        amino_nums_det = len(amino_data_array)
        amino_nums_gt = len(amino_data_gt)

        assert len(amino_data_array) <= self.max_len
        assert len(amino_data_gt) <= self.max_len

        if not self.multi_class:
            # choose to use Binary Label:  Link(1) or Non-Link(0)
            linkage_gt = (linkage_gt != 0).astype(int)
        
        if self.coords_norm:
            amino_data_array = self.coordinates_normalization(amino_data_array)
            amino_data_gt = self.coordinates_normalization(amino_data_gt)

        # also can shuffle after padding
        _amino_data_array = self.add_padding(amino_data_array, _pad_idx=0)
        _amino_data_gt = self.add_padding(amino_data_gt, _pad_idx=0)
        _linkage_gt = self.add_padding(linkage_gt, dims=2)

        if self.using_gt:
            return _amino_data_gt, _linkage_gt, amino_nums_gt

        return _amino_data_array, _linkage_gt, amino_nums_det

    def __len__(self) -> int:
        return len(self.pids)


class MLP_Protein(UniProtein):
    def __init__(self, protein_id, 
                method="concat", # or substract
                max_len=512, block_index=0, re_generate=False,
                fea_src_dir='/mnt/data/zxy/relat_coors_stage3-amino-keypoint-vectors/', 
                label_src_dir='/mnt/data/zxy/stage3_data/stage3_labels/',
                label_linkage_dir='/mnt/data/zxy/stage3_data/stage3_labels/',
                using_whole=False,
                using_gt = False,
                linakge_shuffle=False, dist_rescaling=True, sample_num=None, random_crop=False, crop_bins=10) -> None:
        super().__init__(protein_id, 512, block_index, re_generate, fea_src_dir, label_src_dir, label_linkage_dir, linakge_shuffle, dist_rescaling, sample_num, random_crop, crop_bins)

        self.method = method
        self.concated_data_array = None
        self.subtracted_data_array = None

        if using_gt:
            self._data_array = self.label_data
        else:
            if len(self.data_array) != len(self.label_data):
                self._data_array = self.label_data
            else:
                self._data_array = self.data_array

        # choose a consequent amino-acid as the inpurt source
        if using_whole:
            self.block_len = len(self._data_array)
        else:
            self.block_len =  128                                   # 128 * 128 concatenated arrays will be generated 
        # self.block_len =  512                                     # 512 * 512     #  DY-ATTETNION
        self.data_len = self._data_array.shape[0]
        if self.block_len >= self.data_len:
            self.start_index = 0
        else:
            self.start_index = np.random.randint(0, self.data_len - self.block_len)

        self.mlp_data = self._data_array[self.block_index:self.block_index + self.block_len]
        self.mlp_data_len = len(self.mlp_data)
        self.mlp_linkage = self.linkage_gt_square[self.block_index:self.block_index + self.block_len][:, self.block_index:self.block_index + self.block_len]

        self.processed_data = None
        self.processed_label = None

        self.generate_data()
        self.get_selected_data()

    def generate_data(self):
        """
        Procssed data:
        A.  If method is substract, the data is mainly used to calulate distance
            The "substact method may not be efficient, for the distance of C and N is mutable, not like
            distance between Ca and Ca
        
        B.  Concat
        """
        mlp_data_len = len(self.mlp_data)
        if self.method == "substract":
            temp_data = np.zeros((self.block_len ** 2, 12))
            for i in range(mlp_data_len):
                # 可能会减到自己 ！！
                temp_data[i*mlp_data_len: (i+1)*mlp_data_len] = self.mlp_data[:, 1:] - self.mlp_data[i][1:]
            # labels : 0 or 1 ?
            self.processed_data = temp_data
            self.processed_label = self.mlp_linkage.reshape(-1)
        
        elif self.method == "concat":
            temp_data = np.zeros((self.block_len ** 2, 24))
            for i in range(mlp_data_len):                                                   
                temp_data[i*mlp_data_len: (i+1)*mlp_data_len] = np.concatenate(
                                                                (self.mlp_data[:, 1:], np.tile(self.mlp_data[i][1:], [mlp_data_len, 1])), 
                                                                        axis=1)

            self.processed_data = temp_data
            self.processed_label = self.mlp_linkage.reshape(-1)
            
            #  the second way
            # m = self.block_len
            # n = self.block_len - 1
            # temp_data = np.zeros((self.block_len ** 2, 24))

            # collected_len = 0
            # for i in range(n):
            #     curr_block_len = n - i
            #     temp_data[collected_len : collected_len + curr_block_len] = np.concatenate(
            #                                                                 (self.mlp_data[i+1:, 1:], np.tile(self.mlp_data[i][1:], [curr_block_len, 1])), 
            #                                                                     axis=1)
            #     collected_len += curr_block_len
            #     temp_data[i*mlp_data_len: (i+1)*mlp_data_len] = np.concatenate(
            #                                                     (self.mlp_data[:, 1:], np.tile(self.mlp_data[i][1:], [mlp_data_len, 1])), 
            #                                                             axis=1)
            # self.processed_data = temp_data
            # self.processed_label = self.mlp_linkage.reshape(-1)
        else:
            raise ValueError


    def get_selected_data(self, neg_pos_ratiao=15):
        """
        If choose a block_len = 128, there are around 2 * block_len = 256 positive samples
        And 768 neagative samples will be chose(if the ratio is 3)
        All the sample nums may 1024

        If the ratio = 15: 128 * 16 = 2048 samples generated
        If the ratio = 31: 128 * 32 = 4096 samples generated
        """
        positive_index = np.where(self.processed_label[:self.mlp_data_len**2] != 0)[0]
        neg_index =  np.where(self.processed_label == 0)[0]

        # from negtive sample select 2 * pos_sample_size 
        rand_index = np.random.randint(0, len(neg_index), int(neg_pos_ratiao * len(positive_index)))
        selected_negtives = self.processed_data[neg_index[rand_index]]
        selected_neg_labels = self.processed_label[neg_index[rand_index]]

        selected_postives = self.processed_data[positive_index]
        selected_pos_labels = self.processed_label[positive_index]
        # print(selected_pos_labels, '\n', selected_neg_labels)

        selected_data = np.concatenate([selected_postives, selected_negtives], axis=0)
        selected_label = np.concatenate([selected_pos_labels, selected_neg_labels], axis=0)

        # check and filter 
        if self.method == "substract":
            array_sum = np.sum(selected_data, axis=1)
        elif self.method == "concat":
            array_sum = np.sum(selected_data[:, :12] - selected_data[:, 12:], axis=1)              
        else:
            raise ValueError
        valid_index = (array_sum != 0)
        selected_data = selected_data[valid_index]
        selected_label = selected_label[valid_index]

        self.selected_data = selected_data
        self.selected_label = selected_label

        return None


class MLP_Dataset(Dataset):
    """
        1. get concatenated data
        2. add padding 
        3. set max_len and random sampling
            3.1 The maximum length is limited, for the Memory
        4. Whether to presserve Amino-Type-Info ?
    """
    def __init__(self, index_csv, 
                max_len=4096, pad_idx=-1, 
                fea_src_dir='/mnt/data/zxy/relat_coors_stage3-amino-keypoint-vectors/', 
                label_src_dir='/mnt/data/zxy/stage3_data/stage3_labels/',
                using_gt = False, multi_class=False,
                shuffle=False, dist_rescaling=True,
                coords_norm=False, 
                random_crop=False, crop_bins=10
                ) -> None:
        super().__init__()

        self.index_csv = index_csv
        self.pids = pd.read_csv(index_csv)["p_id"]
        self.pid_visable = False

        self.using_gt = using_gt

        self.max_len = max_len
        self.multi_class = multi_class
        self.shuffle = shuffle
        self.coords_norm = coords_norm
    

    def set_visiable(self, states=False):
        self.pid_visable = states
    
    def add_padding(self, data, _pad_idx=0, dims=1):
        """
            add padding to data and labels 
        """
        _len = len(data)
        if dims == 1:
            result = np.zeros((self.max_len, 1)) + _pad_idx
            result[:_len] = data
            return result
        elif dims == 2:
            result = np.zeros((self.max_len, 24)) + _pad_idx               
            result[:_len] = data
            return result
    
    def coordinates_normalization(self, datas, mean_std_params_file='../datas/params/coord_mean_std.json'):
        length, dims = datas.shape
        assert dims == 24 
        with open(mean_std_params_file, 'r')  as j:
            _params = json.load(j)
        
        x_mean_std = _params['x_mean_std']                  # TODO: Of cource, there face the same problem, the "X" axis and "Z" axis may be disordered
        y_mean_std = _params['y_mean_std']                  # But the mean and std are not far from in X and Z, so, the impact is limmited in some way
        z_mean_std = _params['z_mean_std']                  # DY: ATTENTION
        
        # with out type info
        datas[:, 0::3] = (datas[:, 0::3] - x_mean_std[0])  / x_mean_std[1]
        datas[:, 1::3] = (datas[:, 1::3] - y_mean_std[0])  / y_mean_std[1]
        datas[:, 2::3] = (datas[:, 2::3] - z_mean_std[0])  / z_mean_std[1]

        return datas
    

    def __getitem__(self, index):
        mlp_test = MLP_Protein(protein_id=self.pids[index], method="concat")
        _data = mlp_test.selected_data
        _label = mlp_test.selected_label.reshape(-1, 1)

        # print("Extarcted datlength is {}".format(len(_label)))

        if not self.multi_class:
            # choose to use Binary Label:  Link(1) or Non-Link(0)
            _label = (_label != 0).astype(int)
        
        if self.coords_norm:
            _data = self.coordinates_normalization(_data)

        if self.shuffle:
            rand_index = np.arange(len(_label))
            np.random.shuffle(rand_index)
            _data = _data[rand_index]
            _label = _label[rand_index]

        length = len(_data)
        if length > self.max_len:
            random_index = np.random.randint(0, length, self.max_len)
            selected_data = _data[random_index]
            seletcted_label = _label[random_index]
        else:
            selected_data = self.add_padding(_data, dims=2)
            seletcted_label = self.add_padding(_label, _pad_idx=-1)
        
        return selected_data, seletcted_label

    def __len__(self) -> int:
        return len(self.pids)

    





if __name__ == '__main__':
    train_index_csv='../datas/tracing_data2/train.csv'
    valid_index_csv='../datas/tracing_data2/valid.csv'
    test_index_csv='../datas/tracing_data2/test.csv'
    df1_pids = pd.read_csv(train_index_csv)['p_id'].tolist()
    df2_pids = pd.read_csv(valid_index_csv)['p_id'].tolist()
    df3_pids = pd.read_csv(test_index_csv)['p_id'].tolist()

    pids = df1_pids + df2_pids + df3_pids

    # for pid in pids:
    #     t_protein = UniProtein(protein_id=pid, re_generate=False, linakge_shuffle=True)
    
    # t_protein = UniProtein(protein_id='6PGW', re_generate=False, linakge_shuffle=True)
    t_protein = UniProtein(protein_id='6QDF', re_generate=False, linakge_shuffle=True)
    # 1. Check if the detected array is follow the same orser 
    # Result: YES
    print(t_protein.data_array[:5])
    print("===== ===== =====")
    print(t_protein.label_data[:5])
    detected_ca = t_protein.shuffled_data_array[:][:, 1:4]
    ground_ca = t_protein.shuffled_label_data[:][:, 1:4]
    dist_cnt = []
    for i in range(len(detected_ca)):
        dist_diff = np.array([detected_ca[i][0] - ground_ca[i][2], detected_ca[i][1] - ground_ca[i][1], detected_ca[i][2] - ground_ca[i][0]])
        temp_dist = (dist_diff ** 2).sum() **0.5 * 236
        dist_cnt.append(temp_dist)
        # print("DistDiff: ", temp_dist)
    print("1D mean dist is ", np.mean(dist_cnt))

    # # 2. Chech the shuffle result, if the Ca-Dist is in resonable range
    # # Result: YES
    # shuffled_index_vec = t_protein.rand_index_vec
    # shuffled_gt = t_protein.shuffled_square_gt
    # shuffled_data = t_protein.shuffled_data_array
    # print("Ground Truth Size: ", shuffled_gt.shape)
    # w, h = shuffled_gt.shape
    # for i in range(w):
    #     for j in range(h):
    #         if shuffled_gt[i, j] != 0:
    #             diff = shuffled_data[i][1:] - shuffled_data[j][1:]
    #             print("Ca diff: {}\t Dist: {}".format(diff[:3], ((diff[:3]**2).sum())**0.5 * 236))

    # # 3. Check the C-N dist and N-C dist
    # # Result: YES
    # print("Check CN / NC dist")
    # for i in range(w):
    #         for j in range(h):
    #             if shuffled_gt[i, j] != 0:
    #                 nc_diff = shuffled_data[i][4:7] - shuffled_data[j][7:10]
    #                 cn_diff = shuffled_data[i][7:10] - shuffled_data[j][4:7]

    #                 nc_dist = (nc_diff ** 2).sum() ** 0.5 * 236
    #                 cn_dist = (cn_diff ** 2).sum() ** 0.5 * 236
    #                 if shuffled_gt[i, j] == 1:
    #                     print(i, j, ":LinkageStage: {}\t NC-dist: {}".format(shuffled_gt[i, j], nc_dist))
    #                 else:
    #                     print(i, j, ":LinkageStage: {}\t CN-Dist: {}".format(shuffled_gt[i, j], cn_dist))
    
    # # Check Linkage Dataset
    # test_set = LinkageSet(index_csv='../datas/tracing_data2/test.csv', multi_class=True, using_gt=False, shuffle=True, coors_norm=True,
    #                                                                     random_crop=False, crop_bins=8)
    # test_set.set_visiable(True)
    # test_loader = DataLoader(test_set, shuffle=False, batch_size=1)

    # for idx, batch_data in enumerate(test_loader):
    #     seq_data_array = batch_data[0].to(torch.float32)[0]     #.to(device)
    #     labels = batch_data[1].to(torch.float32)[0]             # batch x seq_len(512) x 13
    #                                                                                     # Need to add a BOS token
    #     amino_nums = batch_data[2]

    #     print(seq_data_array.shape)
    #     print(labels.shape)
    #     print(amino_nums)

    #     print(seq_data_array[:5])
    #     print(seq_data_array[-5:])
    #     print(labels[:5])
    #     print(labels[-5:])
    #     break

    # shuffled_index_vec = t_protein.rand_index_vec
    # shuffled_gt = labels
    # shuffled_data = seq_data_array
    # print("Ground Truth Size: ", shuffled_gt.shape)
    # w, h = shuffled_gt.shape
    # linkage_cntr = 0
    # for i in range(w):
    #         for j in range(h):
    #             if shuffled_gt[i, j] != 0:
    #                 nc_diff = shuffled_data[i][4:7] - shuffled_data[j][7:10]
    #                 cn_diff = shuffled_data[i][7:10] - shuffled_data[j][4:7]
    #                 ca_diff = shuffled_data[i][1:4] - shuffled_data[j][1:4]

    #                 nc_dist = (nc_diff ** 2).sum() ** 0.5# * 236
    #                 cn_dist = (cn_diff ** 2).sum() ** 0.5# * 236
    #                 ca_dist = (ca_diff ** 2).sum() ** 0.5# * 236
    #                 if shuffled_gt[i, j] == 1:
    #                     print(i, j, ":LinkageStage: {}\t NC-dist: {} \tCa-Dist {}".format(shuffled_gt[i, j], nc_dist, ca_dist))
    #                     linkage_cntr += 1
    #                 elif shuffled_gt[i, j] == 2:
    #                     print(i, j, ":LinkageStage: {}\t CN-Dist: {} \tCa-Dist {}".format(shuffled_gt[i, j], cn_dist, ca_dist))
    #                     linkage_cntr += 1
    #                 else:
    #                     pass
    #                     # print(i, j, ": -1, ignore")
    # print("All linkages is ", linkage_cntr)



    # Test data for linear regression
    print("Begin to test MLP data")
    mlp_test = MLP_Protein(protein_id="6QDF", method="concat")
    seletcted_data = mlp_test.selected_data
    seletcted_label = mlp_test.selected_label
    print(seletcted_data)
    print(seletcted_label)

    # check Ca-dist
    # Result: The distance is in the reasonable range 
    # However, check if using the padding data in it        ---- Fixed it
    ca_dists = []
    for item in seletcted_data:
        tmp_ca_dst= ((item[:3] - item[12:15]) ** 2).sum() ** 0.5 * 236
        ca_dists.append(tmp_ca_dst)
    ca_dists = np.array(ca_dists)
    print("Ca dists\n", ca_dists[seletcted_label != 0])
    print("Ca dists\n", ca_dists[seletcted_label == 0])


    # MLP Test Part 
    # Test MLP dataset 
    print("Begin to test mlp_dataset")
    test_set = MLP_Dataset(index_csv='../datas/tracing_data2/test.csv', multi_class=True, using_gt=False, shuffle=True, coords_norm=False,)
    test_set.set_visiable(True)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=1)
    for idx, batch_data in enumerate(test_loader):
        seq_data_array = batch_data[0].to(torch.float32)[0]     #.to(device)
        labels = batch_data[1].to(torch.float32)[0]             # batch x seq_len(512) x 13
                                                                                        # Need to add a BOS token
        print(seq_data_array.shape)
        print(labels.shape)

        print(seq_data_array[:5])
        print(seq_data_array[-5:])
        print(labels[:5])
        print(labels[-5:])
        break 

    shuffled_data = seq_data_array
    shuffled_gt = labels
    print("Ground Truth Size: ", labels.shape)
    w, h = labels.shape
    linkage_cntr = 0
    for i in range(w):
        if labels[i] != 0:
            nc_diff = shuffled_data[i][0:3] - shuffled_data[i][12:15]
            cn_diff = shuffled_data[i][3:6] - shuffled_data[i][15:18]
            ca_diff = shuffled_data[i][6:9] - shuffled_data[i][18:21]

            nc_dist = (nc_diff ** 2).sum() ** 0.5 * 236
            cn_dist = (cn_diff ** 2).sum() ** 0.5 * 236
            ca_dist = (ca_diff ** 2).sum() ** 0.5 * 236

            if shuffled_gt[i] == 1:
                print(i, ":LinkageStage: {}\t NC-dist: {} \tCa-Dist {}".format(shuffled_gt[i], nc_dist, ca_dist))
                linkage_cntr += 1
            elif shuffled_gt[i] == 2:
                print(i, ":LinkageStage: {}\t CN-Dist: {} \tCa-Dist {}".format(shuffled_gt[i], cn_dist, ca_dist))
                linkage_cntr += 1
            else:
                pass
                # print(i, j, ": -1, ignore")
        else:
            nc_diff = shuffled_data[i][0:3] - shuffled_data[i][12:15]
            cn_diff = shuffled_data[i][3:6] - shuffled_data[i][15:18]
            ca_diff = shuffled_data[i][6:9] - shuffled_data[i][18:21]

            nc_dist = (nc_diff ** 2).sum() ** 0.5 * 236
            cn_dist = (cn_diff ** 2).sum() ** 0.5 * 236
            ca_dist = (ca_diff ** 2).sum() ** 0.5 * 236
            print(i, ":LinkageStage: {}\t CN-Dist: {} \tCa-Dist {}".format(shuffled_gt[i], cn_dist, ca_dist))
    print("All linkages is ", linkage_cntr)

