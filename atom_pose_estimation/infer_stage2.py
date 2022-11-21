import os, re, json, sys, math
import numpy as np
import torch
import torch.nn as nn
from layers import HourGlass3DNet
from torch.nn import Module
from scipy.ndimage.interpolation import zoom
import mrcfile
from PIL import Image

import os.path as osp

sys.path.append("..")
from evalutaion import Evaluator
from pdb_utils.pdb_reader_writer import PDB_Reader_Writer, Chain

from amino_acids_tracing.simple_models import MLP
from amino_acids_tracing.Config import Config

AMINO_ACIDS = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
               'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']
AMINO_ACID_DICT = dict(zip(AMINO_ACIDS, range(20)))


def load_model(checkpoint_path):
    pass

def distance(z1, z2, y1, y2, x1, x2):
    z_diff = z1 - z2
    y_diff = y1 - y2
    x_diff = x1 - x2
    sum_squares = math.pow(z_diff, 2) + math.pow(y_diff, 2) + math.pow(x_diff, 2)
    return math.sqrt(sum_squares)


def label_sphere(arr, location, label, sphere_radius):
    box_size = np.shape(arr)
    for x in range(-sphere_radius + location[0], sphere_radius + location[0]):
        for y in range(-sphere_radius + location[1], sphere_radius + location[1]):
            for z in range(-sphere_radius + location[2], sphere_radius + location[2]):
                if (0 <= x < box_size[0] and 0 <= y < box_size[1] and 0 <= z < box_size[2]
                        and distance(location[2], z, location[1], y, location[0], x) < sphere_radius):
                    arr[x][y][z] = label


@torch.no_grad()
def inference(model, data_array, model_params=None):
    """
    Args:
        model: 
        data_array: 
    """
    model.eval()
    Ca_output, N_output, C_output, O_output = model(data_array)
    


def generate_tranformer_data(model, dataset):
    pass


class Stage1_Loader():
    def __init__(self, 
                 hg3d_model,
                 test_pid='6OD0', 
                 mrc_chunks_dir="/mnt/data/zxy/amino-acid-detection/pp_dir/fzw_400_500",
                 detection_result_root_dir="/mnt/data/zxy/amino-acid-detection/pp_dir/detection_pred/",
                 test_protein_dir="/mnt/data/zxy/amino-acid-detection/pp_dir/whole_protein/",
                 infer_test_dir='/mnt/data/fzw/amino-acid-detection/exec_dir/infer_test/',
                 zoom_type="diff") -> None:
        
        self.model = hg3d_model
        self.pdb_id = test_pid
        # self.chunks_dir = mrc_chunks_dir
        self.orgin_mrc_path=None
        self.predictions_dir = os.path.join(infer_test_dir, "pred")
        self.stage1_output_dir = os.path.join(infer_test_dir, "{}_stage1".format(test_pid))
        self.stage2_output_dir = os.path.join(infer_test_dir, "{}_stage2".format(test_pid))
        self.stage3_output_dir = os.path.join(infer_test_dir, "{}_stage3".format(test_pid))
        self.pdb_output_file = os.path.join(infer_test_dir, "{}_result.pdb".format(test_pid))
        # self.pdb_gt_file = "/mnt/data/zxy/amino-acid-detection/EMdata_dir/400_500/{}/simulation/{}.rebuilt.pdb".format(test_pid, test_pid)

        # 0. normalized_mrc and .rebuilt.pdb, cropped chunks
        self.normalized_mrc_file = osp.join(test_protein_dir, test_pid, "normalized_map.mrc")
        self.rebuilt_pdb_file = osp.join(test_protein_dir, test_pid, "{}.rebuilt.pdb".format(test_pid))
        self.pdb_gt_file = osp.join(test_protein_dir, test_pid, "{}.rebuilt.pdb".format(test_pid))
        self.gt_chunks_dir = osp.join(test_protein_dir, test_pid, "chunks")

        # 1. Stage-1 result: amino-acid detection results of each protein, the result is stored by protein-id
        self.detection_result_dir = osp.join(detection_result_root_dir, test_pid)

        # 2. Backbone-reducer: backbone-segmentation file, which store the result fo Backbone-Reducer
        self.backbone_seg_result_file = osp.join(test_protein_dir, test_pid, "prediction", "backbone_reducer_prediction.mrc")
        self.load_backbone_reducer_mrc()

        self.zoom_type=zoom_type
        self.standard_size = [16, 16, 16]
        self.detected_boxs = []

        self.map_size = 236     # fixed setting 
        self.image_dst_dir = './imgs/linkage_images/'

        self.cfg = Config()
        self.load_mrc_offset()
        self.load_linkage_model()
    
    def load_mrc_offset(self, EMdata_dir="/mnt/data/zxy/amino-acid-detection/EMdata_dir/400_500"):
        origin_mrc_file_path = os.path.join(EMdata_dir, self.pdb_id, 'simulation/normalized_map.mrc')
        normalized_map = mrcfile.open(origin_mrc_file_path, mode='r')
        origin_offset = normalized_map.header.origin.item(0)
        map_shape = normalized_map.data.shape
        self.map_origin_offset = origin_offset
        self.map_shape = map_shape

    def parse_chunk_offset(self, chunk_name, core_size=50, r_x=-7+50, start_index=0):
        """
            core_size: chunk 中, 非padding 区域大小
            pad_size: 两侧的padding区域宽度, 与确定chunk的左上角坐标有关
            r_x: index 为 0 的, 对应的, chunk 的 xyz 偏移量, 由于bug, 第 0 个 偏移量设为 43
                 (本应该是 -7 , 与 pad_size 保持一致)
            start_index: 制作数据集的时候, 对于chunk的编号有两个版本: 
                         400_500 文件夹: 一开始 start_index 从 0 开始
                         fzw_400_500:   start_index 从 1 开始 --> 失误
        """
        x_idx, y_idx, z_idx = chunk_name[5:-9].split('_')
        x_offset = (int(x_idx) - start_index) * core_size + r_x #pad_size
        y_offset = (int(y_idx) - start_index) * core_size + r_x #pad_size
        z_offset = (int(z_idx) - start_index) * core_size + r_x #pad_size
        return [x_offset, y_offset, z_offset]
    
    def de_relative_coords(self, real_pos, mrc_offset=None):
        if mrc_offset is None:
            mrc_offset = self.map_origin_offset
        x, y, z = real_pos
        _x = x/2 + mrc_offset[2]
        _y = y/2 + mrc_offset[1]
        _z = z/2 + mrc_offset[0]
        # return np.array([_x, _y, _z])               # TODO: check the coordinates order
        # return [_x, _y, _z]
        return [_z, _y, _x]
    
    def add_padding(self, data, pad_size=2):
        _shape = np.shape(data)
        new_array = np.empty((_shape + 4))

    def load__decteced_boxes(self):
        """
        Only load all detected boxes, aims to:
        1. Compared to original Stage-2 input, 
        2. Do NMS
        3. Reduce coordinates bugs
        """
        # file_names = os.listdir(self.predictions_dir)
        file_names = os.listdir(self.chunks_dir)
        aimed_file_names = []
        for item in file_names:
            if "pred.txt" in item and self.pdb_id in item:
                aimed_file_names.append(item)
        print(aimed_file_names)
        # get corresponding .npy file of chunks
        # eg: 6OD0_0_2_1_pred.txt
        self.gt_coord_files = [(the_name[:-9] + '.txt') for the_name in aimed_file_names]
        self.chunk_npy_files = [(the_name[:-9] + '.npy') for the_name in aimed_file_names]

        detected_boxes = []
        for index, prediction_coord_file in enumerate(aimed_file_names):
            print("Processing {}".format(prediction_coord_file))
            chunk_prections = os.path.join(self.predictions_dir, prediction_coord_file) # txt file
            chunk_array = np.load(os.path.join(self.chunks_dir, self.chunk_npy_files[index]))
            chunk_offsets = self.parse_chunk_offset(prediction_coord_file)
            chunk_offsets = np.array(chunk_offsets)
            print(chunk_offsets, chunk_array.shape)

            with open(chunk_prections, "r") as _preds:
                for line in _preds:
                    # get chunked arrays
                    [x1, x2, y1, y2, z1, z2, a_tpye] = line.split(',')
                    x1, x2, y1, y2, z1, z2, a_tpye = int(x1), int(x2), int(y1), int(y2), int(z1), int(z2), int(a_tpye)
                    detected_boxes.append([x1 + chunk_offsets[0], x2 + chunk_offsets[0], 
                                           y1 + chunk_offsets[1], y2 + chunk_offsets[1],
                                           z1 + chunk_offsets[2], z2 + chunk_offsets[2]])

    def load_backbone_reducer_mrc(self):
        normalized_map = mrcfile.open(self.backbone_seg_result_file, mode='r')
        self.backbone_reducer_map = normalized_map.data
        self.backbone_reducer_offset = normalized_map.header.origin.item(0)

    def get_atom_sphere_backbone_num(self, location, sphere_radius=2):
        box_size = np.shape(self.backbone_reducer_map)
        counter = 0
        location = list(map(int, location))
        for x in range(-sphere_radius + location[0], sphere_radius + location[0]):
            for y in range(-sphere_radius + location[1], sphere_radius + location[1]):
                for z in range(-sphere_radius + location[2], sphere_radius + location[2]):
                    if (0 <= x < box_size[0] and 0 <= y < box_size[1] and 0 <= z < box_size[2]
                        and distance(location[2], z, location[1], y, location[0], x) < sphere_radius):
                        counter += self.backbone_reducer_map[x][y][z]
        return counter

    def load_and_preddict_decteced_boxes(self):
        """
        1. Load cubes from original mrc file
        2. Load cubes from cropped chunks
        """
        # 1. load coordinates file from prediciton dir
        aimed_file_names = os.listdir(self.detection_result_dir)
        aimed_file_names.sort()
        # file_names = os.listdir(self.predictions_dir)
        # aimed_file_names = []
        # for item in file_names:
        #     if "pred.txt" in item and self.pdb_id in item:
        #         aimed_file_names.append(item)
        # print(aimed_file_names)
        # get corresponding .npy file of chunks
        # eg: 6OD0_0_2_1_pred.txt
        gt_file_names = os.listdir(self.gt_chunks_dir)
        self.gt_coord_files = []
        self.chunk_npy_files = []
        for item in gt_file_names:
            if ".txt" in item:
                self.gt_coord_files.append(item)
            else:
                if "backbone" not in item:
                    self.chunk_npy_files.append(item)
        self.gt_coord_files.sort()
        self.chunk_npy_files.sort()
        # self.gt_coord_files = [(the_name[:-9] + '.txt') for the_name in gt_file_names]
        # self.chunk_npy_files = [(the_name[:-9] + '.npy') for the_name in gt_file_names]      # 6OD0_0_2_1.npy
        assert len(aimed_file_names) == len(self.gt_coord_files) == len(self.chunk_npy_files)
        # get cube array, and predict atoms coords
        cube_arrays = []
        predicted_relative_coords = []
        all_chunk_offsets = []
        unnormed_keypoint_predictions = []
        mrc_relates_keypoint_predictions = []
        
        # from IPython import embed; embed()
        # set no_grad 
        self.model.eval()
        with torch.no_grad():
            for index, prediction_coord_file in enumerate(aimed_file_names):
                print("Processing {}".format(prediction_coord_file))

                # load cube detections
                chunk_prections = os.path.join(self.detection_result_dir, prediction_coord_file) # txt file
                
                # load chunked .mrc file
                chunk_array = np.load(os.path.join(self.gt_chunks_dir, self.chunk_npy_files[index]))
                chunk_offsets = self.parse_chunk_offset(prediction_coord_file, start_index=1)
                chunk_offsets = np.array(chunk_offsets)
                print(chunk_offsets, chunk_array.shape)

                with open(chunk_prections, "r") as _preds:
                    for line in _preds:
                        # get chunked arrays
                        [x1, x2, y1, y2, z1, z2, a_tpye, confidence_score] = line.strip().split(',')
                        x1, x2, y1, y2, z1, z2, a_tpye = int(x1), int(x2), int(y1), int(y2), int(z1), int(z2), int(a_tpye)
                        confidence_score = float(confidence_score)
                        # print([x1, y1, z1, x2, y2, z2, a_tpye])
                        # _cube_array = chunk_array[x1:x2, y1:y2, z1:z2]                      # TODO: get data_array with padding=1 ---> chunk_array[x1-1:x2+1, y1-1:y2+1, z1-1:z2+1]
                        # upper_left_point = np.array([x1, y1, z1])

                        t_padding_size = 2
                        _cube_array = chunk_array[max(x1-t_padding_size, 0):min(x2+t_padding_size, 63), 
                                                  max(y1-t_padding_size, 0):min(y2+t_padding_size, 63), 
                                                  max(z1-t_padding_size, 0):min(z2+t_padding_size, 63)]
                        upper_left_point = np.array([max(x1-t_padding_size, 0), max(y1-t_padding_size, 0), max(z1-t_padding_size, 0)])
                        # TODO: get data_array with padding=1 ---> chunk_array[max(x1-1, 0):min(x2+1, 63), max(y1-1, 0):min(y2+1, 63), max(z1-1, 0):min(z2+1, 63)]
                        _cube_size = np.array(_cube_array.shape)
                        cube_arrays.append(_cube_array)

                        # Do prediction
                        _rescaled_array = self.rescale(_cube_array)
                        _rescaled_tensor = torch.from_numpy(_rescaled_array).unsqueeze(0).unsqueeze(0)  # add dimension
                        _rescaled_tensor = _rescaled_tensor.to(torch.float32)
                        Ca_output, N_output, C_output, O_output = model(_rescaled_tensor)               # related coords
                        # print(np.concatenate([Ca_output, N_output, C_output, O_output], axis=1))

                        # check the whether the result is resonable
                        if torch.max(Ca_output) > 1.0 or torch.min(Ca_output) < 0.0:
                            print("Ca Output Error, with value: {}".format(Ca_output))
                        if torch.max(N_output) > 1.0 or torch.min(N_output) < 0.0:
                            print("N Output Error, with value: {}".format(N_output))
                        if torch.max(C_output) > 1.0 or torch.min(C_output) < 0.0:
                            print("C Output Error, with value: {}".format(C_output))
                        if torch.max(O_output) > 1.0 or torch.min(O_output) < 0.0:
                            print("O Output Error, with value: {}".format(O_output))

                        # The coordinages here is related to the [0,  472] size, and related to the UpperLeft 
                        # UpperLeft corner of the re-fomulated .mrc file (density map)
                        abs_Ca = Ca_output[:].detach().cpu().numpy() * _cube_size + upper_left_point + chunk_offsets       # DY: ATTENTION: The [x, y, z] order may distrubed
                        abs_N  = N_output[:].detach().cpu().numpy() * _cube_size  + upper_left_point + chunk_offsets        # abs offsets correlated to the Big-Amino-Aicd File, which size is 272
                        abs_C  = C_output[:].detach().cpu().numpy() * _cube_size  + upper_left_point + chunk_offsets        # 
                        abs_O  = O_output[:].detach().cpu().numpy() * _cube_size  + upper_left_point + chunk_offsets

                        # stage-2 result - relative coords in [236, 236, 236] space
                        # whick is not take the .mrc offset into account 
                        mrc_relate_Ca = self.de_relative_coords(abs_Ca[0], mrc_offset=[0, 0, 0])
                        mrc_relate_N  = self.de_relative_coords(abs_N[0], mrc_offset=[0, 0, 0])
                        mrc_relate_C  = self.de_relative_coords(abs_C[0], mrc_offset=[0, 0, 0])
                        mrc_relate_O  = self.de_relative_coords(abs_O[0], mrc_offset=[0, 0, 0])
                        mrc_relates_keypoint_predictions.append([mrc_relate_Ca, mrc_relate_N, mrc_relate_C, mrc_relate_O])

                        # Stage 2.5: Using backbone reducer result, to determine whether preserver these predicitons
                        Ca_backbone_seg_num = self.get_atom_sphere_backbone_num(mrc_relate_Ca, sphere_radius=2)
                        N_backbone_seg_num  = self.get_atom_sphere_backbone_num(mrc_relate_N, sphere_radius=2)
                        C_backbone_seg_num  = self.get_atom_sphere_backbone_num(mrc_relate_C, sphere_radius=2)
                        O_backbone_seg_num  = self.get_atom_sphere_backbone_num(mrc_relate_O, sphere_radius=2)
                        main_atom_backbone_sum = Ca_backbone_seg_num + N_backbone_seg_num + C_backbone_seg_num + O_backbone_seg_num
                        if main_atom_backbone_sum < 5:
                            print([Ca_backbone_seg_num, N_backbone_seg_num, C_backbone_seg_num, O_backbone_seg_num])
                            print("ATTETNION: May not correct prediction !")
                        # from IPython import embed; embed()

                        # stage-2 result, for test this stage, and generate reports
                        # These coordinates are the final results in .PDB file, so take .mrc offset into account
                        # print([abs_Ca, abs_N, abs_C, abs_O])
                        pdb_absolue_Ca = self.de_relative_coords(abs_Ca[0])
                        pdb_absolue_N  = self.de_relative_coords(abs_N[0])
                        pdb_absolue_C  = self.de_relative_coords(abs_C[0])
                        pdb_absolue_O  = self.de_relative_coords(abs_O[0])
                        unnormed_keypoint_predictions.append([a_tpye] + pdb_absolue_Ca + pdb_absolue_N + pdb_absolue_C + pdb_absolue_O)

                        # convert the coords to the original
                        # for stage3, it need relative coords to the original mrc files
                        s3_fea_vec = np.zeros(13)      # 20  +  4 * 3
                        s3_fea_vec[0] = a_tpye
                        s3_fea_vec[-12:] = np.concatenate([abs_Ca, abs_N, abs_C, abs_O], axis=1) / 236    # DY: 472 here.  ---> May 272 here
                                                                                                                  # for 
                        # print("s3_fea_vec: ", s3_fea_vec)
                        predicted_relative_coords.append(s3_fea_vec.tolist())
                        # predicted_relative_coords = np.concatenate((predicted_relative_coords, s3_fea_vec), axis=1)
        
        # The relative coords and unnormed-coords are of the same order
        predicted_relative_coords = np.array(predicted_relative_coords)
        self.unnormed_keypoint_predictions = np.array(unnormed_keypoint_predictions)
        self.mrc_relates_keypoint_predictions = np.array(mrc_relates_keypoint_predictions)

        self.cube_arrays = cube_arrays
        self.normed_keypoint_predcionts = np.array(predicted_relative_coords)
        print("====  Final: ====")
        print(predicted_relative_coords.shape)
        self.predicted_relative_coords = predicted_relative_coords

    def get_predicted_absolutes_coords(self):
        """
        A subsitute from "load_and_preddict_decteced_boxes"
        """
        print("Using offset: ", self.map_origin_offset)
        unnormed_keypoint_predictions = self.predicted_relative_coords[:]
        unnormed_keypoint_predictions[:, 1:] *= 236  
        print(unnormed_keypoint_predictions[-5:]) 

        unnormed_keypoint_predictions[:, 1::3] = unnormed_keypoint_predictions[:, 1::3] / 2 + self.map_origin_offset[2]      # all X coordinates 
        unnormed_keypoint_predictions[:, 2::3] = unnormed_keypoint_predictions[:, 2::3] / 2 + self.map_origin_offset[1]      # all Y coordinates 
        unnormed_keypoint_predictions[:, 3::3] = unnormed_keypoint_predictions[:, 2::3] / 2 + self.map_origin_offset[0]      # all Z coordinates 

        self.unnormed_keypoint_predcionts = unnormed_keypoint_predictions        # LYS A   1      28.395   9.531  -5.044  1.00  0.00
        print(self.unnormed_keypoint_predcionts[-5:])

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

    def mask_data(self, data_array):
        """
            清除对角线上的输出
        """
        w = data_array.shape[0]
        mask = np.ones((w, w)) - np.eye(w)
        return data_array * mask
    
    def relative_coords_preprocessing(self):
        """
        add normalization to predicted relative Coords: [Ca, N, C, O]
        """
        length = len(self.predicted_relative_coords)
        origin_order = np.arange(length)
        # rand_idx = np.random.shuffle(origin_order)

        # concate the data
        self.predicted_amino_types = self.predicted_relative_coords[:, 0]

        temp_data = np.zeros((length ** 2, 24))
        for i in range(length):                                                   
            temp_data[i*length: (i+1)*length] = np.concatenate((self.predicted_relative_coords[:, 1:], 
                                                                np.tile(self.predicted_relative_coords[i][1:], [length, 1])), axis=1)
        normlized_data = self.coordinates_normalization(temp_data)

        self.concated_data = normlized_data
        print("Concated data shape: ", self.concated_data.shape)

    def load_linkage_model(self, model_params_path="../amino_acids_tracing/ckpnts_dir/2022-0530-1700/model_accu_99.916_recall_99.786.chkpt"):
        mlp_model = MLP(input_dim=24)
        mlp_model = torch.nn.DataParallel(mlp_model)
        mlp_ckpnt = torch.load(model_params_path)
        mlp_state_dict = mlp_ckpnt['model']
        mlp_model.load_state_dict(mlp_state_dict)
        mlp_model.cuda(0)

        self.linakge_model = mlp_model

    def pred_linkage(self):
        _input_array = torch.from_numpy(self.concated_data)
        _input_array = _input_array.to(torch.float32)
        predction = self.linakge_model(_input_array).float()
        predction = torch.sigmoid(predction).detach().cpu().numpy()

        length = int(len(predction) ** 0.5)
        _pred_image = predction.reshape((length, length))
        print("_pred_image shape: ", _pred_image.shape)
        print("_pred_image:", _pred_image)
        _pred_image = self.mask_data(_pred_image)
        print("_pred_image:", _pred_image)

        heatmap_image = (_pred_image * 255).astype(int)
        heatmap_image = Image.fromarray(heatmap_image.astype('uint8'))
        heatmap_image.save(os.path.join(self.image_dst_dir, "{}_heatmap.png".format(self.pdb_id)))
        thres_image  = (_pred_image >= 0.9).astype(int) * 255
        thres_image = Image.fromarray(thres_image.astype('uint8'))
        thres_image.save(os.path.join(self.image_dst_dir, "{}_binary_thres.png".format(self.pdb_id)))


    def fileter(self):
        "Just like do NMS in Object detection, do filtering to elimate some occluded amino acids"
        predicted_relative_coords = self.predicted_relative_coords


                    
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

    def write_result_pdb(self):
        """
        write result to pdb files
        """
        chain = Chain()

        amino_nodes = self.unnormed_keypoint_predictions
        chain.nodes = amino_nodes

        pdb_reader_writer = PDB_Reader_Writer()
        pdb_reader_writer.write_pdb([chain], self.pdb_output_file)
    

    def evalutate(self):
        print("Result evalutation...")
        _input_path = "/home/fzw"
        evaluator = Evaluator(input_path=_input_path)

        result_csv_path = os.path.join(self.predictions_dir, "{}_".format(self.pdb_id))
        print("Prediction file: \t", self.pdb_output_file)
        print("Ground Truth file:\t", self.pdb_gt_file)
        evaluator.evaluate(self.pdb_id, self.pdb_output_file, self.pdb_gt_file, execution_time=10)
        print("Report saved at: ", result_csv_path)
        evaluator.create_report(result_csv_path)
    






if __name__ == '__main__':
    # model = HourGlass3DNet()
    # ckpnt_path = './checkpoints/Hourglass3D_Regression/2022-03-04observation_20.31.58/epoch_95_HG3_CNN.pt'
    # # model = torch.load('./checkpoints/Hourglass3D_Regression/2022-03-07observation_11.00.06/best_HG3_CNN.pt', map_location='cpu')
    # mm  = torch.load(ckpnt_path)
    # # print(type(mm))
    # # nn.Module.load_state_dict(model, torch.load(ckpnt_path))
    # # model.load_state_dict(torch.load(ckpnt_path))
    # # print(model.parameters())
    # model.load_state_dict(mm)
    # print(model.state_dict())

    model = torch.load('./checkpoints/Hourglass3D_Regression/2022-03-07observation_11.00.06/best_HG3_CNN.pt', map_location='cpu')

    stage1_handler = Stage1_Loader(model, test_pid="6OD0")
    # stage1_handler = Stage1_Loader(model, test_pid="6P81")
    # stage1_handler = Stage1_Loader(model, test_pid="6OOB")
    stage1_handler.load_and_preddict_decteced_boxes()
    # stage1_handler.get_predicted_absolutes_coords()

    # from IPython import embed; embed()

    # calculate linkage
    stage1_handler.load_linkage_model()
    stage1_handler.relative_coords_preprocessing()
    stage1_handler.pred_linkage()
    # begin to load 
    stage1_handler.write_result_pdb()
    stage1_handler.evalutate()
