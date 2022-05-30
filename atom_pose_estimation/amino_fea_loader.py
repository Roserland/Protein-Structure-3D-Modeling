"""
    Amino acid feature vector dataset.
    The dataset will use the model trained in "Amino-Pose-Estimation" part.

"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# from cube_loader import AminoAcidDataset, de_normalize
# from atom_pose_estimation.inference import inference
from layers import HourGlass3DNet
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import mrcfile 
import os
from scipy.ndimage.interpolation import zoom
from utils import get_atoms_pos, parse_amino_acid, incre_std_avg
import random
import json



AMINO_ACIDS = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
               'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']
AMINO_ACID_DICT = dict(zip(AMINO_ACIDS, range(20)))
print(AMINO_ACID_DICT)

gpu_id = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
print('gpu ID is ', str(gpu_id))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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



def generate_input_data(model, index_csv='./datas/split/train.csv', save_dir='/mnt/data/zxy/stage3_data/stage3-amino-keypoint-vectors/',
                        absolute=True):
    """
        Using the feature vector produced from Stage-2
        TODO: Whether transoform the REAL coordinates to RELATIVE coordinates with respect to each protein.    ----> It's necessary.

        Args:  
            absulute: if True, after de-normalizing coords, add mrc_file_offset to it, to get the coordinates close to TRUE coord in PDB files.
                      if False, no offset will be added 
        Steps:
        1. Using model trained from stage-2 to get relative coords with a detection cube
        2. Transfer the relatives coords to absolute coords with respect to its original pdb file
        3. Normalize  the coordinates to [0, 1] intercal
        
        4. Get amino type ID, from 1 - 20 for each amino type
        5. For each protein,  add amino id "21" as the EOS of a protein     ----> When create dataset
    """
    model.eval()
    # dataset = AminoAcidDataset(index_csv=index_csv, standard_size=[16, 16, 16],
    #                            gt_type='coords')
    # dataset.set_mode(2)
    # data_loader = DataLoader(dataset)
    df = pd.read_csv(index_csv)
    for i in range(len(df)):
        curr_records = df.iloc[i]
        if absolute:
            _offset = parse_offset(curr_records["offset"])
        else:
            _offset = [0.0, 0.0, 0.0] 
        
        frag_id = curr_records['frag_id']
        mrc_path = curr_records['mrc_path']
        npy_file_path = mrc_path.replace('.mrc', '.npy')     
        amino_type = os.path.basename(mrc_path)[:3]
        amino_index = os.path.basename(mrc_path).strip('.mrc').split('_')[1]        # eg: */GLY_445.mrc

        cube_data_array = np.load(npy_file_path)
        cube_data_array = rescale(cube_data_array)
        cube_data_tensor = torch.from_numpy(cube_data_array)
        cube_data_tensor = cube_data_tensor.unsqueeze(0).unsqueeze(0)                # add dimension
        # cube_data_tensor.to(device)

        npy_basename = os.path.basename(npy_file_path)
        coords_npy_file_path = npy_basename[:3] + '_coord' + npy_basename[3:]
        coords_npy_file_path = npy_file_path.replace(npy_basename, coords_npy_file_path)
        x1, x2, y1, y2, z1, z2, _ = np.load(coords_npy_file_path)
        upper_left_corner = [x1, y1, z1]
        lower_right_corner = [x2, y2, z2]

        # Inference, and prepare feature vector for a SINGLE amino acid
        Ca_output, N_output, C_output, O_output = model(cube_data_tensor)
        pred_Ca = de_normalize(Ca_output[0].detach().cpu().numpy(), upper_left_corner, lower_right_corner, _offset)
        pred_N  = de_normalize(N_output[0].detach().cpu().numpy(), upper_left_corner, lower_right_corner, _offset)
        pred_C  = de_normalize(C_output[0].detach().cpu().numpy(), upper_left_corner, lower_right_corner, _offset)
        pred_O  = de_normalize(O_output[0].detach().cpu().numpy(), upper_left_corner, lower_right_corner, _offset)
        # integrate type and 4 key point atoms positon
        fea_vec = np.zeros(13)      # 20  +  4 * 3
        fea_vec[-12:] = np.concatenate([pred_Ca, pred_N, pred_C, pred_O]) / 236.0       # Why 236 here ---> Its after de_normalize, and the coords has been in [0, 236] already
                                                                                        # 
        fea_vec[0] = AMINO_ACID_DICT[amino_type] + 1
        if((fea_vec[-12:] < 0).sum() > 0 or (fea_vec[-12:] > 1).sum() > 0):
            # break
            print("INVALID Cubes: Frag-ID: {} \t MRC path:".format(frag_id, mrc_path))
            print("Before rescaling: ", fea_vec)
            print("After Rescaling: ", fea_vec[-12:] / 236)
            print("\t")

        # print("Before rescaling: ", fea_vec)
        # print("After Rescaling: ", fea_vec[-12:] / 236)
        # print(fea_vec.shape)
        # print(fea_vec)
        # break

        # save, 
        _save_dir = os.path.join(save_dir, frag_id[:4])
        if not os.path.exists(_save_dir):
            os.makedirs(_save_dir)
        _save_path = os.path.join(_save_dir, os.path.basename(npy_file_path))
        print(_save_path)
        np.save(_save_path, fea_vec)


def generate_input_simulation_data(pdb_index_csv):
    pass


def get_amino_type_from_pdb(pdb_file_path):
    with open(pdb_file_path, 'r') as pdb_file:
        for line in pdb_file:
            if line.startswith("ATOM"):
                amino_acid = parse_amino_acid(line)
                break
    return amino_acid

def get_voxel_pos(pdb_pos, mrc_offset):
    """
    TODO: Check the order of x, y, z
    Because the output data from Stage2-Model may be different from the .pdb ordee, which
    is likely to be [z, y, x] order. 
    Or NOT.
    ATTENTION: Check here !!!
    """
    z = int(pdb_pos[0] - mrc_offset[0]) # * 2
    y = int(pdb_pos[1] - mrc_offset[1]) # * 2
    x = int(pdb_pos[2] - mrc_offset[2]) # * 2
    return [x, y, z]


def generate_output_data(src_dir="/mnt/data/zxy/amino-acid-detection/test_data/pdb_fragments/", 
                         dst_dir='/mnt/data/zxy/stage3_data/stage3_labels/',
                         offset_csv='../datas/mrc_index.csv'):
    """
    Generate an amino-sequence (label) data for each protein;
    The data has following format:
        [[Type, Ca-Pos, N-Pos, C-Pos, O-pos], ...]
    Each row represents a single amino-acid, and the order will be arranged by Proteins' amino-acid-sequence

    Calculate mean_std params for whole dataset(Grond Truth)

    Args: 
        src_dir: directory which save protein datas, each sub-directory in this is named with PROTEIN_ID
        dst_dir: directory which save the OUTPUT data 
        offset_csv: file storing offset of the protein density map;
    IF: Append Amino-Type-Confidence
    """
    """
    1. get offset
    """
    def chunk(_str):
        return _str[:4]
    df = pd.read_csv(offset_csv)
    df['frag_id'] = df['frag_id'].map(chunk)
    df = df.drop_duplicates(subset=['frag_id'])
    df.to_csv("../datas/mrc_offset.csv", index=None)

    offset_dict = {}
    for i in range(len(df)):
        frag_id = df.iloc[i]['frag_id'][:4]
        offset = parse_offset(df.iloc[i]['offset'])
        offset_dict[frag_id] = offset

    protein_ids = sorted(os.listdir(src_dir))
    
    x_pos_std = incre_std_avg()
    y_pos_std = incre_std_avg()
    z_pos_std = incre_std_avg()
    for p_id in protein_ids:
        output_dir = os.path.join(dst_dir, p_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print(output_dir)
        # prepare data for a protein
        curr_dir = os.path.join(src_dir, p_id)
        pdb_files = os.listdir(curr_dir)                    # TODO: 
        index_list = []
        array_list = []
        for pdb_name in pdb_files:
            index_list.append(int(pdb_name.split('.')[0]))
            amino_type= get_amino_type_from_pdb(os.path.join(curr_dir, pdb_name))
            ca, n, c, o = get_atoms_pos(os.path.join(curr_dir, pdb_name))

            feature_vec = [0.0] * 13
            feature_vec[0] = AMINO_ACID_DICT[amino_type] + 1.0
            curr_offset = offset_dict[p_id]
            feature_vec[1:4] = get_voxel_pos(ca, curr_offset)                                   # need add offset
            feature_vec[4:7] = get_voxel_pos(n, curr_offset)            # the coordinates may double
            feature_vec[7:10] = get_voxel_pos(c, curr_offset)
            feature_vec[10:13] = get_voxel_pos(o, curr_offset)
            array_list.append(feature_vec)
        order = np.argsort(np.array(index_list))
        ordred_fea_vecs = np.array(array_list)[order]

        # get relative pos
        ordred_fea_vecs[:, 1:] /= 236.0
        for i in [1, 4, 7, 10]:
            x_pos_std.incre_in_list(ordred_fea_vecs[:, i])
            y_pos_std.incre_in_list(ordred_fea_vecs[:, i+1])
            z_pos_std.incre_in_list(ordred_fea_vecs[:, i+2])
        # print(ordred_fea_vecs[:5])
        np.save(os.path.join(output_dir, "{}.npy".format(p_id)), ordred_fea_vecs)
        np.save(os.path.join(output_dir, "{}_amino_seq_idxs.npy".format(p_id)), np.array(index_list)[order])

        print("x_mean_std: {} {} \ty_mean_std: {} {} \tz_mean_std: {} {} ".format(
            x_pos_std.avg, x_pos_std.std, y_pos_std.avg, y_pos_std.std, z_pos_std.avg, z_pos_std.std, 
        ))
    mean_std_dict = {
        "x_mean_std": [x_pos_std.avg, x_pos_std.std],
        "y_mean_std": [y_pos_std.avg, y_pos_std.std],
        "z_mean_std": [z_pos_std.avg, z_pos_std.std],
    }
    with open('../datas/params/coord_mean_std.json', 'w') as j:
        json.dump(mean_std_dict, j)
    


def generate_data_index(fea_src_dir='/mnt/data/zxy/stage3_data/stage3-amino-keypoint-vectors/', 
                        label_src_dir='/mnt/data/zxy/stage3_data/stage3_labels/',
                        val_split=0.2, output_dir='../datas/tracing_data'
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

        train_idx, test_idx = train_test_split(list(range(len(_train_df))), test_size=0.2)
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


class AminoFeatureDataset(Dataset):
    def __init__(self, index_csv, 
                 fea_src_dir='/mnt/data/zxy/stage3_data/stage3-amino-keypoint-vectors/', 
                 label_src_dir='/mnt/data/zxy/stage3_data/stage3_labels/',
                 max_len = 512, dims = 32, 
                 input_shuffle=True,
                 padding=True, padding_form=None,
                 zoom_type="diff"):
        self.feature_src_dir = fea_src_dir
        self.labels_src_dir = label_src_dir
        self.max_len = max_len
        self.dims = dims
        self.shuffle = input_shuffle
        self.padding_form = padding_form
        
        self.pid = pd.read_csv(index_csv)["p_id"]
    
    def __getitem__(self, index: int):
        pid = self.pid[index]
        protein_fea_dir = os.path.join(self.feature_src_dir, pid)
        protein_label_dir = os.path.join(self.labels_src_dir, pid)
        
        amino_file_list = os.listdir(protein_fea_dir)
        if self.shuffle:
            random.shuffle(amino_file_list)
        data_array = np.zeros([self.max_len, self.dims])
        for i, amino_file in enumerate(amino_file_list):
            data_array[i] = np.load(os.path.join(protein_fea_dir, amino_file)).reshape(-1)
        # add padding
        if self.padding_form is not None:
            for j in range(len(amino_file_list), self.max_len):
                data_array[j] = self.padding_form
        
        label_file = os.path.join(protein_label_dir, "{}.npy".format(pid))
        label_vec = np.load(os.path.join(protein_label_dir, label_file))
        
        print(data_array.shape)
        data_array[:, 0] = data_array[:, 0].astype(int)
        label_vec[:, 0] = np.int(label_vec[:, 0])
        return data_array, label_vec

    def __len__(self) -> int:
        return len(self.pid)


if __name__ =='__main__':
    model = torch.load('./checkpoints/Hourglass3D_Regression/2022-03-07observation_11.00.06/best_HG3_CNN.pt', map_location='cpu')
    # './datas/split/train.csv'  :  store the newy added vector
    # '../datas/split/train.csv' :  the older version, used for backend
    # generate_input_data(model, index_csv='./datas/split/train.csv', save_dir='/mnt/data/zxy/relat_coors_stage3-amino-keypoint-vectors/', absolute=False)
    # generate_input_data(model, index_csv='./datas/split/test.csv',  save_dir='/mnt/data/zxy/relat_coors_stage3-amino-keypoint-vectors/', absolute=False)
    # generate_input_data(model, index_csv='./datas/split/valid.csv', save_dir='/mnt/data/zxy/relat_coors_stage3-amino-keypoint-vectors/', absolute=False)
    # print(model.state_dict())

    # test generated data
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

    generate_output_data()
    generate_data_index()
    pdb_src_dir = '/mnt/data/zxy/stage3_data/stage3_labels/'
    


    # the_dataset = AminoFeatureDataset(index_csv='../datas/tracing_data/train.csv')
    # the_loader  = DataLoader(the_dataset, batch_size=2)
    # for idx, data in enumerate(the_loader):
    #     seq_data_array = data[0].to(torch.float32).to(device)
    #     labels = data[1].to(torch.float32).to(device)
    #     print(seq_data_array)
    #     print(labels)

    #     s1 = seq_data_array[0]
    #     s2 = seq_data_array[1]
    #     print("\nSUM-1: {}".format(s1 == s2).sum())
    #     l1 = labels[0]
    #     l2 = labels[1]
    #     print("\nSUM-2: {}".format(l1 == l2).sum())

