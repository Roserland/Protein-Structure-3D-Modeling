"""
Author: Roserland   2022-04-08

Step 1:
    Just use Ca, N, C, O position vectors, to judge whehter 2 amino-acids could 
    be linked

    A. Construct a Linkage / Affinity Matrix,:
        A.1 Use {Ca, N, C, O} these 4 features rather than just Ca-atom position, 
            aiming to improve the final performance

        A.2 In "Deep Tracer", Dr. Dong Si just use Ca-atom-position, with its 
            amino-acid type and the whole Amino Acid Sequence, to align, or in other
            words, to fit the Seq, 

        A.3 Dong Si tracing the chain before the atom prediction
            He uses TSP algorithm to trace, with Backbone prediction info and Ca atoms
            info, no other infos are needed.

            The Ca atoms are determined by the local peak in the predicted confidence
            map, so it may face the fact that there are many duplicated Ca atoms, in a 
            dense region. Besides, it also requires high precision in the "UNet" 
            segmentation / classification part.

            It is solvable, 



Step 2: A small network, to predict whether 2 amino acids are likely to link?
        Using {Ca, N, C, O} positions

    A. This method, highly depends on the previously predicted result, 

    B. The idea may work, and the dataset is easy to design

    C. It's also possible to add 3D Density map info in it.

Any way, the first step, is critical
"""

"""
    ATTENTION: 

    1. There are many missing sites in a .pdb file, for example in Protein-6XM9: 
        the following residues were not located: 1, 382 - 290, 11 residues missed in total
    
    This case is so usual, so the Transformer Based model may not be robust or efficient.
    
    So when switch to TRY the 'affinity matrix' model, the GT must be generated carefully !!!

"""

import numpy as  np
import os
import random
from PIL import Image
from torch import from_numpy
import matplotlib.pyplot as plt

def ca_dist(ca_pos, rescaling=False):
    """
        ca_pos: N x 3 array
        ca_pos: N x 3 array

    Attention: check the precision
    """

    length, dim = ca_pos.shape
    assert dim == 3
    dist = np.zeros((length, length))

    for i in range(length):
        temp = ca_pos[i] - ca_pos   # using broadcast to calculate
        dist[i] = (temp ** 2).sum(1) ** 0.5
    
    if rescaling:
        dist * 236.0
    
    return dist


class UniProtein():
    def __init__(self, protein_id, max_len=512,
                fea_src_dir='/mnt/data/zxy/relat_coors_stage3-amino-keypoint-vectors/', 
                label_src_dir='/mnt/data/zxy/stage3_data/stage3_labels/',
                shuffle=True, dist_rescaling=True) -> None:
        
        self.protein_id = protein_id
        self.data_dir = os.path.join(fea_src_dir, protein_id)
        self.label_file = os.path.join(label_src_dir, protein_id, "{}.npy".format(protein_id))
        self.label_index_file = os.path.join(label_src_dir, protein_id, "{}_amino_seq_idxs.npy".format(protein_id))

        self.tracked_acid_num = len(os.listdir(self.data_dir))
        # self.gt_acid_num = amino_acids_num
        self.max_len = max_len
        self.shuffle = shuffle
        self.dist_rescaling = dist_rescaling
        self.inkage_gt_square = None
        self.inkage_gt_sets   = None
        
        self.load_tracked_amino_acids()
        self.load_gt_amino_acids()
        self.load_gt_amino_seq_index()
    
    def load_tracked_amino_acids(self, padding=False, pad=None):
        if pad == None:
            pad = -1
        
        if padding:
            data_array = np.zeros((self.max_len, 13)) + pad
        else:
            data_array = np.zeros((self.tracked_acid_num, 13))
        
        amino_file_list = os.listdir(self.data_dir)

        if self.shuffle:
            random.shuffle(amino_file_list)

        # relative coodinates
        for i, amino_file in enumerate(amino_file_list):
            # print(os.path.join(self.data_dir, amino_file))
            data_array[i] = np.load(os.path.join(self.data_dir, amino_file)).reshape(-1)
        
        self.data_array = data_array
        self.amino_types_list = data_array[:, 0]

    def load_gt_amino_acids(self, padding=False, pad=None):
        label_vec = np.load(self.label_file)

        self.gt_amino_num = len(label_vec)
        print("Protein: {}\t Tracked Nums: {}\t GT Nums: {}".format(
            self.protein_id, self.tracked_acid_num, self.gt_amino_num))
        self.label_data = label_vec

    def load_gt_amino_seq_index(self, ):
        index_vec = np.load(self.label_index_file)
        index_vec = np.sort(index_vec)

        exists = np.zeros(index_vec[-1])
        exists[index_vec - 1] = 1
        self.index_vec = index_vec
        missing_idxs = np.where(exists == 0)[0]
        
        self.located_amino_index = index_vec
        self.missing_amino_index = missing_idxs
        self.existence_array = exists

    
    def construct_square_gt(self):
        # index from 1, to the END
        # The end is assumed to be the located amino
        length = self.index_vec[-1]

        gt = np.zeros((length, length))

        for i in range(1, length):
            gt[i, i-1] = 1
            gt[i-1, i] = 1

        for item in self.index_vec:
            if self.existence_array[item - 1] == 0:
                gt[item, item - 1] = 0
                gt[item - 1, item] = 0

        self.inkage_gt_square = gt


    def construct_linkage_set(self):
        linkage_list = []
        length = self.index_vec[-1]

        if self.inkage_gt_square == None:
            self.construct_linkage_set()

        for i in range(1, length):
            if self.inkage_gt_square[i, i-1]:
                linkage_list.append([i, i-1])
        
        self.inkage_gt_sets = linkage_list
        


    @staticmethod
    def cal_ca_dist_matrix(ca_pos, dist_rescaling=True):

        length, dim = ca_pos.shape
        assert dim == 3
        dist = np.zeros((length, length))

        for i in range(length):
            temp = ca_pos[i] - ca_pos   # using broadcast to calculate
            dist[i] = (temp ** 2).sum(1) ** 0.5
    
        if dist_rescaling:
            dist *= 236.0

        return dist
    
    @staticmethod
    def cal_c_n_dist_matrix(c_pos, n_pos, dist_rescaling=True):
        len1, dim1 = c_pos.shape
        len2, dim2 = n_pos.shape

        assert len1 == len2

        dist = np.zeros((len1, len1))

        for i in range(len1):
            temp = c_pos[i] - n_pos
            dist[i] = (temp ** 2).sum(1) ** 0.5

        if dist_rescaling:
            dist *= 236.0
        return dist


    # TODO:
    #   1. check the data generation part, 生成 gt 相对坐标的时候，可能出了问题（距离有一个两倍的关系）
    def get_ca_dist(self, save_dir='./output_dir/affinity_part/imgs'):
        tracked_ca_pos = self.data_array[:, 1:4]
        gt_ca_pos = self.label_data[:, 1:4]
        # print("Relative Tracked max: ", np.max(tracked_ca_pos.reshape(-1)))
        # print("Relative GT max: ", np.max(gt_ca_pos.reshape(-1)))

        tracked_ca_dist_mat = self.cal_ca_dist_matrix(tracked_ca_pos)
        # print("Max in tracked: ", np.max(tracked_ca_dist_mat.reshape(-1)))
        np.fill_diagonal(tracked_ca_dist_mat, 255)
        # print("Min in tracked: ", np.min(tracked_ca_dist_mat.reshape(-1)))

        gt_ca_dist_mat = self.cal_ca_dist_matrix(gt_ca_pos)
        # print(gt_ca_dist_mat)
        # print("Max in gt: ", np.max(gt_ca_dist_mat.reshape(-1)))
        np.fill_diagonal(gt_ca_dist_mat, 255)
        # print("Min in gt: ", np.min(gt_ca_dist_mat.reshape(-1)))

        img_dir = os.path.join(save_dir, self.protein_id)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        tracked_img = Image.fromarray(tracked_ca_dist_mat.astype('uint8'))
        gt_img = Image.fromarray(gt_ca_dist_mat.astype('uint8'))
        tracked_img.save(os.path.join(img_dir, "{}_track_image.png").format(self.protein_id))
        gt_img.save(os.path.join(img_dir, "{}_gt_image.png").format(self.protein_id))

        _thres_tracked = (tracked_ca_dist_mat <= 4).astype(int) * 255
        _thres_gt = (gt_ca_dist_mat <= 4.2).astype(int) * 255
        # print("gt linkages", _thres_gt.reshape(-1).sum() / 255 / 2)
        _thres_tracked_img = Image.fromarray(_thres_tracked.astype('uint8'))
        _thres_gt_img = Image.fromarray(_thres_gt.astype('uint8'))
        _thres_tracked_img.save(os.path.join(img_dir, "{}_thres_tracked_image.png").format(self.protein_id))
        _thres_gt_img.save(os.path.join(img_dir, "{}_thres_gt_img.png").format(self.protein_id))

        gt_neighbor_ca_dists = []
        for i in range(0, len(gt_ca_pos) - 1):
            gt_neighbor_ca_dists.append(gt_ca_dist_mat[i, i+1])
        gt_neighbor_ca_dists = np.array(gt_neighbor_ca_dists)

        self.gt_neighbor_ca_dists = gt_neighbor_ca_dists

        # print(min(gt_neighbor_ca_dists), max(gt_neighbor_ca_dists))
        # print("May the missing num: ", (gt_neighbor_ca_dists > 5).sum(),  "length ", len(gt_neighbor_ca_dists))
        # print((gt_neighbor_ca_dists - 3.8).sum() / 3.5)

    
    def get_C_N_dist(self, save_dir='./output_dir/affinity_part/imgs'):
        """
            check distance between the prev-amino's C and the next-amino's N
        """
        # print("\nCalculating aminos' C-N atom distance ")

        tracked_n_pos = self.data_array[:, 4:7]
        tracked_c_pos = self.data_array[:, 7:10]
        # tracked_c_pos = self.data_array[:, 10:13]
        gt_n_pos = self.label_data[:, 4:7]
        gt_c_pos = self.label_data[:, 7:10]
        # gt_c_pos = self.label_data[:, 10:13]

        img_dir = os.path.join(save_dir, self.protein_id)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        tracked_c_n_dist = self.cal_c_n_dist_matrix(tracked_c_pos, tracked_n_pos)
        # np.fill_diagonal(tracked_c_n_dist, 255)
        gt_c_n_dist = self.cal_c_n_dist_matrix(gt_c_pos, gt_n_pos)
        # np.fill_diagonal(gt_c_n_dist, 255)
        tracked_img = Image.fromarray(tracked_c_n_dist.astype('uint8'))
        gt_img = Image.fromarray(gt_c_n_dist.astype('uint8'))
        tracked_img.save(os.path.join(img_dir, "{}_C_N_track_image.png").format(self.protein_id))
        gt_img.save(os.path.join(img_dir, "{}_C_N_gt_image.png").format(self.protein_id))

        _thres_tracked = (tracked_c_n_dist <= 2.2).astype(int) * 255
        _thres_gt = (gt_c_n_dist <= 2.2).astype(int) * 255
        # print("gt linkages", _thres_gt.reshape(-1).sum() / 255 / 2)
        _thres_tracked_img = Image.fromarray(_thres_tracked.astype('uint8'))
        _thres_gt_img = Image.fromarray(_thres_gt.astype('uint8'))
        _thres_tracked_img.save(os.path.join(img_dir, "{}_thres_C_N_tracked_image.png").format(self.protein_id))
        _thres_gt_img.save(os.path.join(img_dir, "{}_thres_C_N_gt_img.png").format(self.protein_id))

        gt_neighbor_cn_dists = []
        for i in range(0, len(gt_c_pos) - 1):
            gt_neighbor_cn_dists.append(gt_c_n_dist[i, i+1])
        gt_neighbor_cn_dists = np.array(gt_neighbor_cn_dists)

        self.gt_neighbor_cn_dists = gt_neighbor_cn_dists

        # print(min(gt_neighbor_cn_dists), max(gt_neighbor_cn_dists))
        # print("Ma``````````y the missing num: ", (gt_neighbor_cn_dists > 2).sum(),  "length ", len(gt_neighbor_cn_dists))
        # print((gt_neighbor_cn_dists - 3.8).sum() / 3.5)


def get_protein_params(EM_data_dir = "/mnt/data/zxy/amino-acid-detection/EMdata_dir/400_500",
                       save_dir = './output_dir/affinity_part/dist_statics',
                       save_data=True):
    protein_ids = os.listdir(EM_data_dir)

    gt_ca_dist_list = np.array([])
    gt_cn_dist_list = np.array([])
    for i, pid in enumerate(protein_ids):
        uni_protein = UniProtein(pid)
        uni_protein.get_ca_dist()
        uni_protein.get_C_N_dist()
        gt_ca_dist_list = np.concatenate((gt_ca_dist_list, uni_protein.gt_neighbor_ca_dists))
        gt_cn_dist_list = np.concatenate((gt_cn_dist_list, uni_protein.gt_neighbor_cn_dists))

    
    print("\n**** Ca distance distribution ****")
    print("Max: {}\t Min: {}\t Mean: {}".format(np.max(gt_ca_dist_list), np.min(gt_ca_dist_list), np.mean(gt_ca_dist_list)))

    print("\n**** CN distance distribution ****")
    print("Max: {}\t Min: {}\t Mean: {}".format(np.max(gt_cn_dist_list), np.min(gt_cn_dist_list), np.mean(gt_cn_dist_list)))

    if save_data:
        np.save(os.path.join(save_dir, "ca_dist.npy"), gt_ca_dist_list)
        np.save(os.path.join(save_dir, "cn_dist.npy"), gt_cn_dist_list)

    


def get_protein_params_from_original_pdb(EM_data_dir = "/mnt/data/zxy/amino-acid-detection/EMdata_dir/400_500"):
    protein_ids = os.listdir(EM_data_dir)
    
    aim_pid = '6XM9'

    pdb_fragments_dir = "/mnt/data/zxy/amino-acid-detection/test_data/pdb_fragments/"

    cnt_arr = [0] * (489+1)
    for item in os.listdir(pdb_fragments_dir + aim_pid):
        idx = int(item[:-4])
        cnt_arr[idx] = 1
    cnt_arr = np.array(cnt_arr)

    missing_idx = np.where(cnt_arr == 0)[0]
    print("missing amino seq indices are: ", missing_idx)
        

def plot_dist_distribution(npy_file_path, save_path, phase, thres_max=5, _type='bin'):
    data = np.load(npy_file_path)
    print("Original {} distribution: qualified length is {}".format(phase, len(data)))
    data = data[data <= thres_max]
    print("{} distribution: qualified length is {}".format(phase, len(data)))
    if _type == "bin":
        plt.hist(data, bins=40)
    plt.title("{} dist distribution plot".format(phase))

    plt.savefig(save_path)
    plt.close()



if __name__ == "__main__":
    a = np.random.rand(4, 3)

    # print(a)

    # dist = ca_dist(a)
    # print(dist)

    protein_a = UniProtein(protein_id="6XM9")
    print(protein_a.index_vec)
    print(protein_a.existence_array)
    # protein_a.get_ca_dist()
    # protein_a.get_C_N_dist()

    # get_protein_params_from_original_pdb()

    # get_protein_params()



    # # ###  Begin -- Temp Test
    # print("\n******** Test Part ********")
    # ca_489 = np.array([4.201, 22.842, 61.339])
    # ca_488 = np.array([4.128, 19.076, 60.841])

    # c_134 = np.array([13.747, 9.954, 33.622])
    # n_135 = np.array([14.751, 10.405, 32.781])

    # c_488 = np.array([4.599, 20.522, 60.813])
    # n_489 = np.array([3.695, 21.424, 61.187])

    # dist = ((ca_489 - ca_488) ** 2).sum() ** 0.5
    # dist2 = ((c_488 - n_489) ** 2).sum() ** 0.5
    # print("Test dist: ", dist,  '\t',  dist2, "\n")
    # # ###  End ---- Temp Test





    # #### Begin ----  Plot distribution 
    # plot_dist_distribution(npy_file_path="./output_dir/affinity_part/dist_statics/ca_dist.npy", 
    #                        save_path='./output_dir/affinity_part/dist_statics/ca_dist.png',
    #                        thres_max=5.0,
    #                        phase="Ca-Ca")
    # plot_dist_distribution(npy_file_path="./output_dir/affinity_part/dist_statics/cn_dist.npy", 
    #                        save_path='./output_dir/affinity_part/dist_statics/cn_dist.png',
    #                        thres_max=2.5,
    #                        phase="C-N")
    # #### End   ----  Plot distribution 
