import imp
from operator import mod
import torch
import torch.nn as nn
from layers import HourGlass3DNet
from torch.nn import Module

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


if __name__ == '__main__':
    model = HourGlass3DNet()
    ckpnt_path = './checkpoints/Hourglass3D_Regression/2022-03-04observation_20.31.58/epoch_95_HG3_CNN.pt'
    mm  = torch.load(ckpnt_path)
    # print(type(mm))
    # nn.Module.load_state_dict(model, torch.load(ckpnt_path))
    # model.load_state_dict(torch.load(ckpnt_path))
    # print(model.parameters())
    model.load_state_dict(mm)
    print(model.state_dict())

