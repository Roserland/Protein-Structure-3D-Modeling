import torch
import mrcfile
import os, json
import os.path as osp
import h5py
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader
from collections import deque
from models.UNet3D import UNet3D
import math
from utils.map_splitter import reconstruct_map, create_manifest
from rmsd import rmsd_score,load_data,probable_ca_mask,distance
from postprocessing.pdb_reader_writer import *
import argparse
import pandas as pd
from collections import Counter


def remove_small_chunks(input_image):
    """A method used to remove small disjoint regions in a 3D image
    This was primarily used to clean up the backbone prediction .MRC file
    however it may not be necessary
    """
    min_chuck_size = 25
    box_size = np.shape(input_image)
    visited = np.zeros(box_size)
    for x in range(1, box_size[0] - 1):
        for y in range(1, box_size[1] - 1):
            for z in range(1, box_size[2] - 1):
                if input_image[x, y, z] > 0 and visited[x, y, z] == 0:
                    chunk_list = list()
                    queue = deque()
                    queue.append([x, y, z])
                    chunk_list.append([x, y, z])
                    while len(queue) > 0:
                        cur_position = queue.popleft()
                        visited[cur_position[0], cur_position[1], cur_position[2]] = 1
                        offsets = [[0, 0, -1], [0, 0, 1], [0, -1, 0], [0, 1, 0], [-1, 0, 0], [1, 0, 0]]
                        for index in range(len(offsets)):
                            x_new = cur_position[0] + offsets[index][0]
                            y_new = cur_position[1] + offsets[index][1]
                            z_new = cur_position[2] + offsets[index][2]
                            if 0 <= x_new < box_size[0] and 0 <= y_new < box_size[1] and 0 <= z_new < box_size[2]:
                                if input_image[x_new, y_new, z_new] > 0 and visited[x_new, y_new, z_new] == 0:
                                    queue.append([x_new, y_new, z_new])
                                    visited[x_new, y_new, z_new] = 1
                                    chunk_list.append([x_new, y_new, z_new])
                    if len(chunk_list) < min_chuck_size:
                        for voxel in chunk_list:
                            input_image[voxel[0], voxel[1], voxel[2]] = 0




def prediction_and_visualization(model_save_path, normalized_map, save_path, pdb_id):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet3D(
            in_channels=1, 
            out_channels=args.output_channel, 
            final_sigmoid=False, 
            f_maps=16, layer_order='cr',
            num_groups=8, 
            num_levels=5, 
            is_segmentation=True, conv_padding=1)
    
    model = model.to(device)
    if torch.cuda.device_count() > 1 :
        model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_save_path),False)
    model.eval()
    normalized_map = mrcfile.open(normalized_map, mode='r')

    full_image = normalized_map.data
    manifest = create_manifest(full_image)
    prediction_image = np.zeros((np.shape(manifest)))
   

    for index in range(math.ceil(len(manifest) / 10)):
        input = manifest[index * 10: (index + 1) * 10]
        input = torch.tensor(input).unsqueeze(dim=1)
        input = input.to(device,dtype=torch.float32)
        output = model(input)  
        output = torch.softmax(output, dim=1) 
        output1 = output[:, 1:2, :, :, :].squeeze(dim=1).cpu().detach().numpy()
        #output0 = output[:,0:1,:,:,:].squeeze(dim=1).cpu().detach().numpy()
        #prediction = np.subtract(output1,output0)
        prediction_image[index * 10: (index + 1) * 10] = output1

    #prediction_image += 10

    prediction_image = reconstruct_map(prediction_image, np.shape(full_image))
    input_mask = np.where(full_image > 0, 1, 0)
    prediction_image = np.where(input_mask == 1, prediction_image, 0)
    #backbone_image[backbone_image < 0] = 0
    prediction_image = np.array(prediction_image, dtype=np.float32)


    #remove_small_chunks(backbone_image)

    with mrcfile.new(save_path, overwrite=True) as mrc:
        mrc.set_data(prediction_image)
        mrc.header.origin = normalized_map.header.origin.item(0)
        mrc.update_header_stats()
        mrc.close()


def prediction_and_visualization_amino_acid(model_save_path, normalized_map, save_path, pdb_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
    model = UNet3D(in_channels=1, out_channels=21, final_sigmoid=False, f_maps=16, layer_order='cr',
                num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1)
    model = model.to(device)
    if torch.cuda.device_count() > 1 :
        model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_save_path),False)
    model.eval()
    normalized_map = mrcfile.open(normalized_map, mode='r')

    full_image = normalized_map.data
    manifest = create_manifest(full_image)
    prediction_image = np.zeros((np.shape(manifest)))

    for index in range(math.ceil(len(manifest) / 10)):
        input = manifest[index * 10: (index + 1) * 10]
        input = torch.tensor(input).unsqueeze(dim=1)
        input = input.to(device,dtype=torch.float32)
        output = model(input)  
        
        output1 = output[:,1:,:,:,:]
        output1 = torch.argmax(output1, dim=1).squeeze(dim=1).cpu().detach().numpy()
        
        #output0 = output[:,0:1,:,:,:].squeeze(dim=1).cpu().detach().numpy()
        #prediction = np.subtract(output1,output0)
        prediction_image[index * 10: (index + 1) * 10] = output1


    prediction_image = reconstruct_map(prediction_image, np.shape(full_image))
    input_mask = np.where(full_image > 0, 1, 0)
    prediction_image = np.where(input_mask == 1, prediction_image, 0)
    #backbone_image[backbone_image < 0] = 0
    prediction_image = np.array(prediction_image, dtype=np.float32)


    #remove_small_chunks(backbone_image)

    with mrcfile.new(save_path, overwrite=True) as mrc:
        mrc.set_data(prediction_image)
        mrc.header.origin = normalized_map.header.origin.item(0)
        mrc.update_header_stats()
        mrc.close()


def prediction_and_visualization_backbone(model_save_path, normalized_map, save_path, args, pdb_id):
    """
    Visualization of Backbone
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet3D(
        in_channels=1, 
        out_channels=args.output_channel, 
        final_sigmoid=False, 
        f_maps=16, 
        layer_order='cr',
        num_groups=8, 
        num_levels=5, 
        is_segmentation=True, 
        conv_padding=1)

    model = model.to(device)
    if torch.cuda.device_count() > 1 :
        model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load(model_save_path),False)
    model.eval()
    normalized_map = mrcfile.open(normalized_map, mode='r')

    full_image = normalized_map.data
    manifest = create_manifest(full_image)
    prediction_image = np.zeros((np.shape(manifest)))

    for index in range(math.ceil(len(manifest) / 10)):
        input = manifest[index * 10: (index + 1) * 10]
        input = torch.tensor(input).unsqueeze(dim=1)
        input = input.to(device,dtype=torch.float32)
        output = model(input)  
        output = torch.softmax(output, dim=1) 
        
        #output1 = output[:,1:,:,:,:]
        output1 = output[:, 1:2, :, :, :].squeeze(dim=1).cpu().detach().numpy()
        
        #output1 = torch.argmax(output1, dim=1).squeeze(dim=1).cpu().detach().numpy()
        
        #output0 = output[:,0:1,:,:,:].squeeze(dim=1).cpu().detach().numpy()
        #prediction = np.subtract(output1,output0)
        prediction_image[index * 10: (index + 1) * 10] = output1

    from IPython import embed; embed()
    prediction_image = reconstruct_map(prediction_image, np.shape(full_image))
    input_mask = np.where(full_image > 0, 1, 0)
    prediction_image = np.where(input_mask == 1, prediction_image, 0)
    #backbone_image[backbone_image < 0] = 0
    prediction_image = np.array(prediction_image, dtype=np.float32)


    #remove_small_chunks(backbone_image)

    with mrcfile.new(save_path, overwrite=True) as mrc:
        mrc.set_data(prediction_image)
        mrc.header.origin = normalized_map.header.origin.item(0)
        mrc.update_header_stats()
        mrc.close()


def get_chunk_offset(chunk_name):
    xyz = chunk_name[5:10]
    x, y, z = map(int, xyz.split('_'))
    return x, y, z

def get_manifest(chunk_names, chunks_dir):
    manifest = []
    for chunk in chunk_names:
        chunk_path = osp.join(chunks_dir, chunk)
        chunk_data = np.load(chunk_path)



def prediction_and_visualization_backbone_reducer(
        model_save_path, args, pdb_id, 
        proteins_dir="/mnt/data/zxy/amino-acid-detection/pp_dir/whole_protein"):
    """
    Get backbone reducer .mrc results. If need .pdb results, the post-processing of transfering segmentation 
    to atom-postion is needed.

    1. Get prediction of each chunk
    2. reconstruct the whole protein with its chunks
    3. each chunk is named by its croppd position when do the pre-processing.

    chunke-size: 64 x 64 x 64
    core-size: 50 * 50 * 50
    """
    # preparations, data_dir
    p_dir = osp.join(proteins_dir, pdb_id)
    chunks_dir = osp.join(p_dir, "chunks")
    chunk_names = os.listdir(chunks_dir)
    chunk_names.sort()
    chunk_names = list(filter(lambda x: "backbone" not in x, chunk_names))

    print("prediction_and_visualization_backbone_reducer: check sort")

    # save path
    save_dir = osp.join(p_dir, 'prediction')
    os.makedirs(save_dir, exist_ok=True)
    save_path = osp.join(save_dir, "backbone_reducer_prediction.mrc")

    # get normalized map
    normalized_map = osp.join(p_dir, "normalized_map.mrc")
    normalized_map = mrcfile.open(normalized_map, mode='r')
    full_image = normalized_map.data
    origin = normalized_map.header.origin.item(0)           # offset
    
    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet3D(
        in_channels=1, 
        out_channels=args.output_channel, 
        final_sigmoid=False, 
        f_maps=16, 
        layer_order='cr',
        num_groups=8, 
        num_levels=5, 
        is_segmentation=True, 
        conv_padding=1)

    model = model.to(device)
    if torch.cuda.device_count() > 1 :
        model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load(model_save_path),False)
    model.eval()

    manifest = create_manifest(full_image)
    prediction_image = np.zeros((np.shape(manifest)))

    for index in range(math.ceil(len(manifest) / 10)):
        input = manifest[index * 10: (index + 1) * 10]
        input = torch.tensor(input).unsqueeze(dim=1)
        input = input.to(device,dtype=torch.float32)
        output = model(input)  
        output = torch.softmax(output, dim=1) 
        
        output1 = output[:, 1:2, :, :, :].squeeze(dim=1).cpu().detach().numpy()

        prediction_image[index * 10: (index + 1) * 10] = output1
    
    prediction_image = reconstruct_map(prediction_image, np.shape(full_image))
    input_mask = np.where(full_image > 0, 1, 0)
    prediction_image = np.where(input_mask == 1, prediction_image, 0)
    #backbone_image[backbone_image < 0] = 0
    prediction_image = np.array(prediction_image, dtype=np.float32)

    with mrcfile.new(save_path, overwrite=True) as mrc:
        mrc.set_data(prediction_image)
        mrc.header.origin = normalized_map.header.origin.item(0)
        mrc.update_header_stats()
        mrc.close()


def rmsd_result(csv_path):
    df = pd.read_csv(csv_path,sep = '\t')
    print(df.mean())
    df['sub'] = df['# Native Ca Atoms']-df['# Modeled Ca Atoms']
    print(df[df['sub']>0])



def eval_reducer_with_detection(protein_dirs="/mnt/data/zxy/amino-acid-detection/pp_dir/whole_protein"):
    """
    Ablation study of backbone-reducer
    """
    pids = os.listdir(protein_dirs)
    file_names = os.listdir()


if __name__ == '__main__':

    """
    Usage :
    python prediction.py    --dataset_path /mnt/data/Storage4/mmy/CryoEM_0112_train  \
                            --amino_acid_model_path ./checkpoints/0112_0921_amino_acid_simulation_checkpoint_epoch_8   \
                            --atom_model_path ./checkpoints/0112_0920_atom_simulation_checkpoint_epoch_29   \
                            --type simulation   \
                            --overwrite
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--amino_acid_model_path', type=str, required=True)
    parser.add_argument('--atom_model_path', type=str, required=True)
    parser.add_argument('--type', type=str, required=True)
    parser.add_argument('--overwrite', action='store_const', const=True, default=False)
    parser.add_argument('--output_channel', default=2, type=int)

    args = parser.parse_args()
    

    amino_acids = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
                   'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']


    test_dir = args.dataset_path
    filenames = os.listdir(test_dir)
    amino_acid_model_save_path = args.amino_acid_model_path
    atom_model_save_path = args.atom_model_path
    df = pd.DataFrame()
    for pdb_id in filenames:
        print(pdb_id)
        normalized_map = os.path.join(test_dir, pdb_id, args.type, 'normalized_map.mrc')
        if os.path.exists(normalized_map):
            atom_save_path = os.path.join(test_dir, pdb_id, args.type, 'atom_prediction.mrc')
            acid_save_path = os.path.join(test_dir, pdb_id, args.type, 'amino_acid_prediction.mrc')
            if(args.overwrite or not (os.path.exists(atom_save_path) and os.path.exists(acid_save_path))):
                print("saving: ", atom_save_path)
                prediction_and_visualization_backbone(atom_model_save_path, normalized_map, atom_save_path, args, pdb_id)
                # prediction_and_visualization_amino_acid(amino_acid_model_save_path ,normalized_map,acid_save_path,pdb_id)
            
            gt_pdb_file_path = os.path.join(test_dir, pdb_id,pdb_id + '_ca.pdb')
            pred_pdb_file_path = os.path.join(test_dir, pdb_id)
            rmsd_score(atom_save_path, acid_save_path, gt_pdb_file_path, pdb_id, pred_pdb_file_path)
            data = pd.read_excel('/home/fzw/Cryo-EM/Protein-Structure-3D-Modeling/ca_filter/tmp/{}_outputresults.xls'.format(pdb_id))
            df = df.append(data[0:1])

    df.to_csv('./tmp/'+args.type+'_result.csv',sep = '\t')
    #rmsd_result('./tmp/'+args.type+'_result.csv')
    

            
            

            

