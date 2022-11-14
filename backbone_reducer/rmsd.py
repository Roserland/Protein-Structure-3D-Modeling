import os
import mrcfile
import numpy as np
from collections import Counter
from postprocessing.pdb_reader_writer import *
from copy import deepcopy
import evaluation

def distance(z1, z2, y1, y2, x1, x2):
    """Calculates Euclidean distance between two points"""
    z_diff = z1 - z2
    y_diff = y1 - y2
    x_diff = x1 - x2
    sum_squares = np.power(z_diff, 2) + np.power(y_diff, 2) + np.power(x_diff, 2)
    return np.sqrt(sum_squares)


def sphere_mask(radius=4.0):
    h_w = np.ceil(radius).astype(int)
    length = int(1 + h_w * 2)
    mask = np.zeros([length, length, length])
    for i in range(length):
        for j in range(length):
            for k in range(length):
                dist = np.sqrt((h_w - i) ** 2 + (h_w - j) ** 2 + (h_w - k) ** 2)
                # print("coords: {}, dist:{}".format([i, j, k], dist))
                if dist <= radius:
                    mask[i, j, k] = 1
                else:
                    pass
    return mask

def split_chains(backbone_image, thres=0.5):
    """
    Used to identify disconnected chains;
    By rounding each voxel of the confidence map to either zero or one and
    then found connected areas of voxels with a value of one
    ATTENTION: This method required that the out put of the Backbone_Image must be in the [0, 1] interval
    However: The output of baseline model is not in the [0, 1] interval
    Args:
        backbone_image: backbone image predicted by the model
        thres:          判断为 backbone 部分的阈值
    Returns:
        labels: 形状和类型与 backbone_image 一致, 连通的区域使用相同的整数标记
        nums  : 标签数, 等于最大索引标签, 仅在return_num为True的时候返回
    """
    from skimage import measure, color
    # rounded_image = np.round(backbone_image).astype(int)
    rounded_image = (backbone_image >= thres).astype(int)
    labels, num = measure.label(rounded_image, return_num=True)
    if num < 1:
        print("This image can not be specified")
        raise ValueError
    return labels, num

def load_data(file_path="/Users/fanzw/PycharmProjects/cryoem-tianchi/outputs_cnn/0559/cnn/backbone_confidence.mrc"):
    data = mrcfile.open(file_path, mode='r')
    # data = data.header.origin.item(0)
    # print("voxel size:", data.voxel_size)
    return deepcopy(data.data), data.header.origin.item()

def calculate_weighted_center(data_array):
    """
    calculate a weighted center_pos of data
    Args:
        data_array: 3D array
    Returns:  a average weighted position
    """
    assert len(data_array.shape) == 3
    weighted_pos = np.zeros(3)
    counter = 0
    w, h, l = data_array.shape
    weights = 0.0
    for i in range(w):
        for j in range(h):
            for k in range(l):
                confidence_value = data_array[i, j, k]
                if confidence_value > 0.3:
                    weighted_pos += np.array([i, j, k]) * confidence_value
                    counter += 1
                    weights += confidence_value
                else:
                    pass
    # return weighted_pos / counter
    return weighted_pos / weights


#def probable_ca_mask(ca_image, backbone_image, radius=3, ):
def probable_ca_mask(ca_image, radius=3, ):
    """
    Step 1: find all probable Ca atoms position among ca_confidence_image predicted by CNN.
            They are all local maximums in the confidence map within a distance of four voxels
            that have a minimum value of 0.5.
    Methods: 1. find all local maximums, within a cube
             2. if the confidence value of this local maximum is < 0.5, reject
                else: restore its index
    Args:
        ca_image:       np.array, 3D array,
        backbone_image: used to get average density values between two Ca atoms, value is One or Zero
        radius:         a distance prefixed by user/paper
    Tips: a valid Ca atom, must lie in the backbone_image
    TODO: 1. dots ca_image and backbone_ima ge to make sure all found Ca atoms are in backbone.
          2. using two floating point coordinates to calculates average density values between 2 Ca atoms
          3. if method 2. is difficult to work, try anything?
    Ques: 1. Why the value of ca_confidence image is not in the interval [0, 1]?
             Max value of ca_confidence image is 22.19747
    For Deep Tracer Methods, the Radius might be 2.0, with backbone-image multiplied
    Returns:
    """
    sphere = sphere_mask(radius=radius)[:-1, :-1, :-1]  # eg, from [9, 9, 9] to [8, 8, 8]
    #sphere = sphere_mask(radius=radius)
    #print("Mask shape is {}".format(sphere.shape))
    rad_int = np.ceil(radius).astype(int)
    w, l, h = ca_image.shape
    ca_pos_grid = list()
    ca_pos_floating = list()
    diffs = []
    #_ca_image = ca_image * backbone_image
    _ca_image = ca_image 
    for i in range(rad_int, w - rad_int):
        for j in range(rad_int, l - rad_int):
            for k in range(rad_int, h - rad_int):
                local_confidence_value = _ca_image[i, j, k]
                if local_confidence_value < 0.9:###非常重要的变量
                    continue
                aim_cube = _ca_image[i - rad_int: i + rad_int, j - rad_int: j + rad_int, k - rad_int: k + rad_int]
                tmp_cube = sphere * aim_cube
                if local_confidence_value < np.max(tmp_cube):
                    # not the local maximum
                    continue
                # TODO: 
                offset = np.array([i- rad_int, j- rad_int, k- rad_int])  # - np.array([rad_int, rad_int, rad_int])  # add offsets
                tmp_weighted_pos = calculate_weighted_center(tmp_cube)
                # print("Offset([i, j ,k]): {} \t calculated diff: {}".format(offset, tmp_weighted_pos))
                tmp_weighted_pos += offset
                # print("Offset([i, j ,k]): {} \t calculated pos: {}".format(offset, tmp_weighted_pos))
                ca_pos_grid.append(np.array([i, j, k]))
                ca_pos_floating.append(tmp_weighted_pos)
                diffs.append(tmp_weighted_pos - np.array([i, j, k]))
                # sphere[i, j, k] = 1     # mask
                # set used gas to zero
                # _ca_image[i - rad_int: i + rad_int, j - rad_int: j + rad_int, k - rad_int: k + rad_int] = (1 - sphere) * aim_cube
    #print("local_maximum nums is :{}".format(len(ca_pos_grid)))
    ret = {
        "ca_grid_pos": ca_pos_grid,
        "ca_floating_pos": ca_pos_floating,
    }
    return ret


def rmsd_score(mrc_path, aa_mrc_path, pdb_file, pdb_id, out_pdb_path='./tmp/', evaluate=True):
    ca_confidence_image, offset1 = load_data(file_path=mrc_path)
    
    amino_acids = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
               'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']
    #mask = sphere_mask(radius=2)
    # 0. load data
    #ca_confidence_image, offset1 = load_data(
    #    file_path="/home/mmy/CA-BACKBONE_PREDCTION_v2/2CU5_1229_atom_test.mrc")
    #backbone_image, offset2 = load_data(
    #    file_path="/home/mmy/CA-BACKBONE_PREDCTION_v2/2CU5_1228_test.mrc")
    #print("offsets: {} -- {}".format(offset1, offset2))

    # 1. Identifying / Splitting chains
    #image_labels, label_nums = split_chains(backbone_image)
    #label_cntr = Counter(image_labels.reshape(-1))
    #print(label_cntr)

    # 2. get valid chain labels
    #valid_labels = []
    #for key in label_cntr.keys():
    #    if label_cntr[key] > 10 and key is not max(label_cntr, key=lambda x: label_cntr[x]):
    #        # 大于10, 且不为最大值对应的key
    #        valid_labels.append(key)
    #print("Valid Labels: {}".format(valid_labels))

    # 3. find local maximum as Ca atoms(gas)
    # just use (label == 1) as test
    # TODO: solve the inconsistent problem between num-of-local-maximum and num-of-aimed-Ca-atom-in-pdb
    # In fact, the predicted Ca-Atoms are almost in the backbone area
    #chain_1 = (image_labels == 1).astype(int)
    #chain_1 = chain_1 * backbone_image
    #backbone_image[backbone_image < 0.7] = 0
    ca_sets = probable_ca_mask(ca_image=ca_confidence_image)
    ca_grid_chain_1 = ca_sets['ca_grid_pos']                        # the Ca-Postion are int
    ca_float_chain_1 = ca_sets['ca_floating_pos']                   # Ca-Positions are weighted and calculated into float 
    # print(ca_grid_chain_1[:10], ca_float_chain_1[:10])

    # 3.1 try to write a pdb to see whether the prediction is OK
    # print("In rmsd.py: -1 ")
    # from IPython import embed; embed()
    ca_float_chain_1 = np.array(ca_float_chain_1)
    the_nodes = np.array([ca_float_chain_1[:, 2]/2, ca_float_chain_1[:, 1]/2, ca_float_chain_1[:, 0]/2]).swapaxes(0, 1)
    my_chain = Chain()

    if(os.path.exists(aa_mrc_path)):
        aa_confidence_image, offset1 = load_data(
        file_path=aa_mrc_path)
        aa_list = []
       
        for ca in ca_float_chain_1:
            index_array = []
            for x in range(int(ca[0]-1),int(ca[0]+2)):
                for y in range(int(ca[1]-1),int(ca[1]+2)):
                    for z in range(int(ca[2]-1),int(ca[2]+2)):
                        if(distance(x,ca[0],y,ca[1],z,ca[2])<2):
                            index_array.append(aa_confidence_image[int(x),int(y),int(z)])
            ##          
            index_array = np.array(index_array,dtype = int)
            aa_index = np.argmax(np.bincount(index_array))
            aa_list.append(amino_acids[int(aa_index)])
            my_chain.amino_acids = aa_list
    
    # fixed_bias = [0.25, 0.25, 0.25]
    fixed_bias = [0.25, 0.25, 0.25]     # Why these values ?
    my_chain.nodes = the_nodes  + np.array([offset1[0]+fixed_bias[0], offset1[1]+fixed_bias[0], offset1[2]+fixed_bias[0]])
    test_pdb_file_name = out_pdb_path+'/'+pdb_id+'_output.pdb'
    pdb_writer = PDB_Reader_Writer()
    pdb_writer.write_pdb(chains=[my_chain], file_name=test_pdb_file_name)
    if(evaluate):
        evaluator = evaluation.Evaluator('..')
        evaluator.evaluate(pdb_id, test_pdb_file_name, pdb_file, 120)
        evaluator.create_report('/home/fzw/Cryo-EM/Protein-Structure-3D-Modeling/ca_filter/tmp/{}_output'.format(pdb_id), 120)
        evaluator.print_report()


    # 4. calculate distance cost between these nodes, establish linkage matrix between nodes.
    #valid_nodes_list, cost_linkage_matrix = establish_ca_graph(ca_pos_list=ca_grid_chain_1,
    #                                                           backbone_image=backbone_image,
    #                                                          density_map=ca_confidence_image)
    #print(cost_linkage_matrix.shape)
    #print(cost_linkage_matrix[:10, :10])

    # 5. using TSP method to trace Ca-Chains
    #print("*** TSP Tracing... ***")
    #tsp_tracing(ca_link_graph=cost_linkage_matrix, ca_pos_list=valid_nodes_list)


if __name__ == '__main__':

    
    #backbone_image, offset2 = load_data(
    #    file_path="/home/mmy/CA-BACKBONE_PREDCTION_v2/2CU5_1228_test.mrc")
    #print("offsets: {} -- {}".format(offset1, offset2)) 

    pdb_id='6H9S'
    rmsd_score("/home/mmy/CryoEM/tmp/6H9S_0_ca_test.mrc",'/mnt/data_2/mmy/CryoEM_0112_test/6H9S/6H9S_ca.pdb', pdb_id)


    