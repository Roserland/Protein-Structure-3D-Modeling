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
import pandas as pd
import numpy as np
import mrcfile 
import os
from scipy.ndimage.interpolation import zoom
from utils import get_atoms_pos, parse_amino_acid



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



def generate_input_data(model, index_csv='../datas/split/train.csv', save_dir='/mnt/data/zxy/stage3_data/stage3-amino-keypoint-vectors/'):
    """
        Args: 使用 Stage2 中产生的向量作为 Input
        TODO: 是否将 坐标 转换成 （相应）蛋白质空间下的相对坐标,  eg. x 的范围为 [0, 236) 或 [0, 472) （0.5 voxel分辨率下)
    """
    model.eval()
    # dataset = AminoAcidDataset(index_csv=index_csv, standard_size=[16, 16, 16],
    #                            gt_type='coords')
    # dataset.set_mode(2)
    # data_loader = DataLoader(dataset)
    df = pd.read_csv(index_csv)
    for i in range(len(df)):
        curr_records = df.iloc[i]
        _offset = parse_offset(curr_records["offset"])
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
        fea_vec = np.zeros(32)      # 20  +  4 * 3
        fea_vec[-12:] = np.concatenate([pred_Ca, pred_N, pred_C, pred_O])
        fea_vec[AMINO_ACID_DICT[amino_type]] = 1.0

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


def generate_output_data(src_dir="/mnt/data/zxy/amino-acid-detection/test_data/pdb_fragments/", 
                         dst_dir='/mnt/data/zxy/stage3_data/stage3_labels/'):
    """
    Generate an amino-sequence (label) data for each protein;
    The data has following format:
        [[Type-One-Hot-Code, Ca-Pos, N-Pos, C-Pos, O-pos], ...]
    Each row represents a single amino-acid, and the order will be arranged by Proteins' amino-acid-sequence

    Args: 
        src_dir: directory which save protein datas, each sub-directory in this is named with PROTEIN_ID
        dst_dir: directory which save the OUTPUT data 
    """
    protein_ids = sorted(os.listdir(src_dir))
    for p_id in protein_ids:
        output_dir = os.path.join(dst_dir, p_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print(output_dir)
        # prepare data for a protein
        curr_dir = os.path.join(src_dir, p_id)
        pdb_files = os.listdir(curr_dir)
        index_list = []
        array_list = []
        for pdb_name in os.listdir(curr_dir):
            index_list.append(int(pdb_name.split('.')[0]))
            amino_type= get_amino_type_from_pdb(os.path.join(curr_dir, pdb_name))
            ca, n, c, o = get_atoms_pos(os.path.join(curr_dir, pdb_name))

            feature_vec = [0.0] * 32
            feature_vec[AMINO_ACID_DICT[amino_type]] = 1.0
            feature_vec[20:23] = ca
            feature_vec[23:26] = n
            feature_vec[26:29] = c
            feature_vec[29:32] = o
            array_list.append(feature_vec)
        order = np.argsort(np.array(index_list))
        ordred_fea_vecs = np.array(array_list)[order]
        # print(ordred_fea_vecs[:5])
        np.save(os.path.join(output_dir, "{}.npy".format(p_id)), ordred_fea_vecs)


            


        




class AminoFeatureDataset(Dataset):
    def __init__(self, index_csv, 
                 standard_size=[32, 32, 32],
                 gt_type=None,
                 zoom_type="diff"):
                 pass

if __name__ =='__main__':
    model = torch.load('./checkpoints/Hourglass3D_Regression/2022-03-07observation_11.00.06/best_HG3_CNN.pt', map_location='cpu')
    # model.to(device)
    # generate_input_data(model, index_csv='../datas/split/train.csv', save_dir='/mnt/data/zxy/stage3-amino-keypoint-vectors/')
    # generate_input_data(model, index_csv='../datas/split/test.csv', save_dir='/mnt/data/zxy/stage3-amino-keypoint-vectors/')
    # generate_input_data(model, index_csv='../datas/split/valid.csv', save_dir='/mnt/data/zxy/stage3-amino-keypoint-vectors/')
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

