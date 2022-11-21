import numpy as np
import pandas as pd 
import os, sys
import os.path as osp
import mrcfile

"""
    Try to re-evaluate the backbone-reducer. 

    Experiment A: 
        1. Using Detection boxes to calculate average Backbone-Ca-Density 
        2. Using GT bboxes to calculate averaget backbone-ca-density
    
    How to get CORRECT COORDINATES?  See From "./atom_pose_estimation/infer_stage2.py: Stage1_Loader
"""

def parse_chunk_offset(chunk_name, core_size=50, chunk_size=64):
    x, y, z = map(int, chunk_name[5:].split())

def parse_chunk_offset(chunk_name, core_size=50, r_x=-7+50):
    """
        core_size: chunk 中, 非padding 区域大小
        pad_size: 两侧的padding区域宽度, 与确定chunk的左上角坐标有关
        r_x: index 为 0 的, 对应的, chunk 的 xyz 偏移量, 由于bug, 第 0 个 偏移量设为 43
             (本应该是 -7 , 与 pad_size 保持一致)
    """
    x_idx, y_idx, z_idx = chunk_name[5:10].split('_')
    x_offset = int(x_idx) * core_size + r_x #pad_size
    y_offset = int(y_idx) * core_size + r_x #pad_size
    z_offset = int(z_idx) * core_size + r_x #pad_size
    return [x_offset, y_offset, z_offset]


class ReducerEvaluator:
    def __init__(self, pid, binary_thres=0.35,
                gt_bboxes_dir="/mnt/data/zxy/amino-acid-detection/pp_dir/whole_protein/",
                pred_bboxes_dir="/mnt/data/zxy/amino-acid-detection/pp_dir/detection_pred/") -> None:
        self.binary_thres = binary_thres
        self.backbone_reducer_prediction_file = osp.join(gt_bboxes_dir, pid, 
                                                        "prediction", "backbone_reducer_prediction.mrc")
        self.gt_bboxes_dir = osp.join(gt_bboxes_dir, pid, "chunks")
        self.detected_bboxes_dir = osp.join(pred_bboxes_dir, pid)
        self.gt_pdb_path = osp.join(gt_bboxes_dir, pid, "{}.rebuilt.pdb".format(pid))
        self.normalized_mrc_map_path = osp.join(gt_bboxes_dir, pid, "normalized_map.mrc")
    
        self.gt_bboxes_files = get_valid_file_list(self.gt_bboxes_dir)
        self.pred_bboxes_files = get_valid_file_list(self.detected_bboxes_dir)

        self.gt_bboxes = self.load_all_gt_bboxes_coords(self.gt_bboxes_dir, load_type='GT')
        self.pred_bboxes = self.load_all_gt_bboxes_coords(self.detected_bboxes_dir, load_type='Pred')

        self.load_backbone_reducer()

    @staticmethod
    def load_all_gt_bboxes_coords(anno_dir, load_type="GT"):
        valid_files = get_valid_file_list(anno_dir)

        bboxes_list = []
        scores_list = []
        class_list  = []
        for item in valid_files:
            
            chunk_offsets = parse_chunk_offset(item)
            # chunk_offsets = np.array(chunk_offsets)

            file_path = osp.join(anno_dir, item)
            with open(file_path, 'r') as f:
                file_data = f.readlines()
                for row in file_data:
                    tmp_data = row.split(',')
                    if load_type == 'GT':
                        x1, x2, y1, y2, z1, z2, amino_class = map(int, tmp_data)
                        score = 1.0
                    else:
                        x1, x2, y1, y2, z1, z2, amino_class = map(int, tmp_data[:-1])
                        score = float(tmp_data[-1])
                    # These coordinates are relative to the normalized_mrc file, so there is 
                    # no need to transfer to ABSOLUTE coodinates corresponding to PDB files.
                    bboxes_list.append([x1 + chunk_offsets[0], x2 + chunk_offsets[0], 
                                        y1 + chunk_offsets[1], y2 + chunk_offsets[1],
                                        z1 + chunk_offsets[2], z2 + chunk_offsets[2]])
                    scores_list.append(score)
                    class_list.append(amino_class)
        
        res_dict = {
            "bboxes": bboxes_list,
            "scores": scores_list,
            "class":  class_list,
            "data_type": load_type
        }

        return res_dict

    def load_backbone_reducer(self):
        reducer_normalized_map = mrcfile.open(self.backbone_reducer_prediction_file, mode='r')
        self.normalized_offset = reducer_normalized_map.header.origin.item(0)           # offset
        self.reducer_normalized_map = reducer_normalized_map.data

    def sum_backbone_pixel_per_cube(self):
        pred_bboxes = self.pred_bboxes['bboxes']

        pred_backbone_pixel_counter = []
        for i, bbox in enumerate(pred_bboxes):
            x1, x2, y1, y2, z1, z2 = bbox
            backbone_part = self.reducer_normalized_map[x1:x2, y1:y2, z1:z2]
            # from IPython import embed; embed()
            backbone_voxel_num = (backbone_part > self.binary_thres).sum()    
            pred_backbone_pixel_counter.append(backbone_voxel_num)
        print("Pred: ", len(pred_backbone_pixel_counter), self.binary_thres)
        print(pred_backbone_pixel_counter)
        print("\n")
        gt_bboxes = self.gt_bboxes['bboxes']
        gt_backbone_pixel_counter = []
        for i, bbox in enumerate(gt_bboxes):
            x1, x2, y1, y2, z1, z2 = bbox
            backbone_part = self.reducer_normalized_map[x1:x2, y1:y2, z1:z2]
            # from IPython import embed; embed()
            backbone_voxel_num = (backbone_part > self.binary_thres).sum()    
            gt_backbone_pixel_counter.append(backbone_voxel_num)
        print("GT:  ", len(gt_backbone_pixel_counter), self.binary_thres)
        print(gt_backbone_pixel_counter)


def get_valid_file_list(bboxes_dir):
    """
    Args:
        bboxes dir: gt / prediction bboxes
    """
    def _filter(name_str):
        return '.txt' in name_str
    file_names = list(filter(_filter, os.listdir(bboxes_dir)))
    file_names.sort()

    return file_names

if __name__ == '__main__':
    evaluator = ReducerEvaluator("6OOB", binary_thres=0.25)
    gt_bboxs = evaluator.load_all_gt_bboxes_coords(
        evaluator.gt_bboxes_dir, load_type='GT'
    )
    # print(gt_bboxs)

    pred_bboxes = evaluator.load_all_gt_bboxes_coords(
        evaluator.detected_bboxes_dir, load_type='Pred'
    )
    # print(pred_bboxes)

    evaluator.sum_backbone_pixel_per_cube()


