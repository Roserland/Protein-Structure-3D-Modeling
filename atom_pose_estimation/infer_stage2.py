from cgi import test
import imp
from operator import mod
import os, re
import numpy as np
import torch
import torch.nn as nn
from layers import HourGlass3DNet
from torch.nn import Module
from scipy.ndimage.interpolation import zoom
import mrcfile

from evalutaion import Evaluator
from pdb_utils.pdb_reader_writer import PDB_Reader_Writer, Chain

AMINO_ACIDS = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
               'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']
AMINO_ACID_DICT = dict(zip(AMINO_ACIDS, range(20)))


def load_model(checkpoint_path):
    pass

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
                 mrc_chunks_dir="/mnt/data/zxy/amino-acid-detection/pp_dir/400_500",
                 infer_test_dir='/mnt/data/fzw/amino-acid-detection/exec_dir/infer_test/',
                 zoom_type="diff") -> None:
        
        self.model = hg3d_model
        self.pdb_id = test_pid
        self.chunks_dir = mrc_chunks_dir
        self.orgin_mrc_path=None
        self.predictions_dir = os.path.join(infer_test_dir, "pred")
        self.stage1_output_dir = os.path.join(infer_test_dir, "{}_stage1".format(test_pid))
        self.stage2_output_dir = os.path.join(infer_test_dir, "{}_stage2".format(test_pid))
        self.stage3_output_dir = os.path.join(infer_test_dir, "{}_stage3".format(test_pid))
        self.pdb_output_file = os.path.join(infer_test_dir, "{}_result.pdb".format(test_pid))
        self.pdb_gt_file = "/mnt/data/zxy/amino-acid-detection/EMdata_dir/400_500/{}/simulation/{}.rebuilt.pdb".format(test_pid, test_pid)

        self.zoom_type=zoom_type
        self.standard_size = [16, 16, 16]
        self.detected_boxs = []

        self.map_size = 236     # fixed setting 

        self.load_mrc_offset()
    
    def load_mrc_offset(self, EMdata_dir="/mnt/data/zxy/amino-acid-detection/EMdata_dir/400_500"):
        origin_mrc_file_path = os.path.join(EMdata_dir, self.pdb_id, 'simulation/normalized_map.mrc')
        normalized_map = mrcfile.open(origin_mrc_file_path, mode='r')
        origin_offset = normalized_map.header.origin.item(0)
        map_shape = normalized_map.data.shape
        self.map_origin_offset = origin_offset
        self.map_shape = map_shape

    def parse_chunk_offset(self, chunk_name, core_size=50, r_x=-7+50):
        """
            core_size: chunk 中, 非padding 区域大小
            pad_size: 两侧的padding区域宽度, 与确定chunk的左上角坐标有关
            r_x: index 为 0 的, 对应的, chunk 的 xyz 偏移量, 由于bug, 第 0 个 偏移量设为 43
                 (本应该是 -7 , 与 pad_size 保持一致)
        """
        x_idx, y_idx, z_idx = chunk_name[5:-9].split('_')
        x_offset = int(x_idx) * core_size + r_x #pad_size
        y_offset = int(y_idx) * core_size + r_x #pad_size
        z_offset = int(z_idx) * core_size + r_x #pad_size
        return [x_offset, y_offset, z_offset]
    
    def de_normlize_coords(self, real_pos):
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
        file_names = os.listdir(self.predictions_dir)
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


    def load_and_preddict_decteced_boxes(self):
        """
        1. Load cubes from original mrc file
        2. Load cubes from cropped chunks
        """
        # 1. load coordinates file from prediciton dir
        file_names = os.listdir(self.predictions_dir)
        aimed_file_names = []
        for item in file_names:
            if "pred.txt" in item and self.pdb_id in item:
                aimed_file_names.append(item)
        print(aimed_file_names)
        
        # get corresponding .npy file of chunks
        # eg: 6OD0_0_2_1_pred.txt
        self.gt_coord_files = [(the_name[:-9] + '.txt') for the_name in aimed_file_names]
        self.chunk_npy_files = [(the_name[:-9] + '.npy') for the_name in aimed_file_names]

        # get cube array, and predict atoms coords
        cube_arrays = []
        predicted_relative_coords = []
        all_chunk_offsets = []
        unnormed_keypoint_predictions = []
        
        # set no_grad 
        self.model.eval()
        with torch.no_grad():
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

                        # stage-2 result, for test this stage, and generate reports
                        # print([abs_Ca, abs_N, abs_C, abs_O])
                        de_normed_Ca = self.de_normlize_coords(abs_Ca[0])
                        de_normed_N  = self.de_normlize_coords(abs_N[0])
                        de_normed_C  = self.de_normlize_coords(abs_C[0])
                        de_normed_O  = self.de_normlize_coords(abs_O[0])
                        unnormed_keypoint_predictions.append([a_tpye] + de_normed_Ca + de_normed_N + de_normed_C + de_normed_O)


                        # convert the coords to the original

                        # for stage3, it need relative coords to the original mrc files
                        s3_fea_vec = np.zeros(13)      # 20  +  4 * 3
                        s3_fea_vec[0] = a_tpye
                        s3_fea_vec[-12:] = np.concatenate([abs_Ca, abs_N, abs_C, abs_O], axis=1) / 236    # DY: 472 here.  ---> May 272 here
                                                                                                                  # for 
                        # print("s3_fea_vec: ", s3_fea_vec)
                        predicted_relative_coords.append(s3_fea_vec.tolist())
                        # predicted_relative_coords = np.concatenate((predicted_relative_coords, s3_fea_vec), axis=1)

        predicted_relative_coords = np.array(predicted_relative_coords)

        self.cube_arrays = cube_arrays
        self.normed_keypoint_predcionts = np.array(predicted_relative_coords)
        print("====  Final: ====")
        print(predicted_relative_coords.shape)
        self.predicted_relative_coords = predicted_relative_coords

        self.unnormed_keypoint_predictions = np.array(unnormed_keypoint_predictions)
    
    def get_predicted_absolutes_coords(self):
        print("Using offset: ", self.map_origin_offset)
        unnormed_keypoint_predictions = self.predicted_relative_coords[:]
        unnormed_keypoint_predictions[:, 1:] *= 236  
        print(unnormed_keypoint_predictions[-5:]) 

        unnormed_keypoint_predictions[:, 1::3] = unnormed_keypoint_predictions[:, 1::3] / 2 + self.map_origin_offset[2]      # all X coordinates 
        unnormed_keypoint_predictions[:, 2::3] = unnormed_keypoint_predictions[:, 2::3] / 2 + self.map_origin_offset[1]      # all Y coordinates 
        unnormed_keypoint_predictions[:, 3::3] = unnormed_keypoint_predictions[:, 2::3] / 2 + self.map_origin_offset[0]      # all Z coordinates 

        self.unnormed_keypoint_predcionts = unnormed_keypoint_predictions        # LYS A   1      28.395   9.531  -5.044  1.00  0.00
        print(self.unnormed_keypoint_predcionts[-5:])

        
    def pred():
        pass

                    
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
        # n_chain = Chain()
        # c_chain = Chain()
        # o_chain = Chain()

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
        evaluator.create_report(result_csv_path)
    

    def fileter():
        pass






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
    stage1_handler.load_and_preddict_decteced_boxes()
    # stage1_handler.get_predicted_absolutes_coords()
    stage1_handler.write_result_pdb()
    stage1_handler.evalutate()