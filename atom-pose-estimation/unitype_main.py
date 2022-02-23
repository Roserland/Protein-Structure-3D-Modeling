#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py    
@Contact :   roserland@gmail.com
Try to use 3D U-Net / 3D CNN to regress some 'Key Points' among an amino acid cube, such as:
    1. The Ca atom, which will be used in Protein Backbone Tracing
    2. The N atom in NH2, will be used to calculate Ca-N-CO angle, to determine amino acid cube's spatial position.
    3. The O atom in CO,
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/22 17:20 下午   fanzw      1.0         None
'''

"""
This file is a elementary test about uni-type amino-acid position predcition, we will:
    0. Using uni-type amino acid cube to train.
    1. First, use 3D-UNet to regress a single atom coordinates: [x, y, z]
    2. Second, use some separate model to predict their position among such as 'Ca, N, C, O' atoms 
    3. Finally, using a single model to predict these 3 or 4 atoms' position, for there must be some spatial connection
       between these atoms.
"""

# import lib
from pkg_resources import parse_requirements
from traceback2 import print_tb
from light_3DUNet import My_3D_CNN
from cube_loader import *
from atom_pred_visualize import plot_coords_diffs
from torch.optim import Adam, lr_scheduler
import os, time, datetime
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import argparse

AMINO_ACIDS = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
               'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']


parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=str, required=True)
parser.add_argument("--amino_type", type=str, required=True)
parser.add_argument("--batch_size", default=256)
args = parser.parse_args()

# prepare environment
gpu_id = args.gpu_id
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
print('gpu ID is ', str(gpu_id))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current Device:", device)

LR = 0.01
w_decay = 0.001
lr_decay_step_size = 10
lr_decay_ratio = 0.2
batch_size = args.batch_size
n_epochs = 120
loss_weight = [1, 1.0, 1.0, 1.0]

log_dir = './log_files/'
f_l1_loss_log = open(os.path.join(log_dir, 'key_point_pos_estimation_loss.txt'),  'a')


def train(epoch, model, train_loader, optimizer, _scheduler):
    model.train()
    train_loss = 0
    n_batches = 0
    for batch_idx, data in enumerate(train_loader):
        curr_time = time.time()
        cube_data_array = data[0].to(torch.float32).to(device)
        Ca_pos = data[1].to(torch.float32).to(device)
        N_pos = data[2].to(torch.float32).to(device)
        C_pos = data[3].to(torch.float32).to(device)
        O_pos = data[4].to(torch.float32).to(device)
        # print(C_pos)
        optimizer.zero_grad()

        Ca_output, N_output, C_output, O_output = model(cube_data_array)
        loss_Ca = loss_fn(Ca_output, Ca_pos)
        loss_N = loss_fn(N_output, N_pos)
        loss_C = loss_fn(C_output, C_pos)
        loss_O = loss_fn(O_output, O_pos)
        loss = loss_weight[0] * loss_Ca + loss_weight[1] * loss_N + \
               loss_weight[2] * loss_C + loss_weight[3] * loss_O

        loss.backward()
        optimizer.step()
        # print(Ca_output)
        used_time = round(time.time() - curr_time, 4)
        print("[Batch:{}--{}s] \tCa_L1:{}\tN_L1:{}\tC_L1:{}\tO_L1:{}\tloss:{}".format(
            batch_idx, used_time, loss_Ca, loss_N, loss_C, loss_O, loss)
        )

        train_loss += loss.item()
        n_batches += 1
        if n_batches % 20 == 0:      # 20
            break

    train_loss = train_loss / n_batches
    print("Epoch {} Train loss: {}".format(epoch, train_loss))
    writelog(f_l1_loss_log, "Train loss: {}".format(train_loss))


def test(epoch, phase, model, test_loader, optimizer, _scheduler):
    model.eval()
    test_loss = 0.0
    n_batches = 0.0
    print("[Test/Valid] -- [Epoch: {}]".format(epoch))

    gt_Ca = []; pred_Ca = []
    gt_N  = []; pred_N = []
    gt_C = []; pred_C = []
    gt_O = []; pred_O = []
    for batch_idx, data in enumerate(test_loader):
        curr_time = time.time()
        cube_data_array = data[0].to(torch.float32).to(device)
        Ca_pos = data[1].to(torch.float32).to(device)
        N_pos = data[2].to(torch.float32).to(device)
        C_pos = data[3].to(torch.float32).to(device)
        O_pos = data[4].to(torch.float32).to(device)
        # print(C_pos)
        gt_Ca += data[5][:].numpy().tolist() # .to(torch.float32).to(device)
        gt_N  += data[6][:].numpy().tolist() # .to(torch.float32).to(device)
        gt_C  += data[7][:].numpy().tolist() # .to(torch.float32).to(device)
        gt_O  += data[8][:].numpy().tolist() # .to(torch.float32).to(device)
        # de_normalize
        upper_left_corner = data[9][:].numpy()
        lower_right_corner = data[10][:].numpy()
        _offset = data[11][:].numpy()


        Ca_output, N_output, C_output, O_output = model(cube_data_array)
        loss_Ca = loss_fn(Ca_output, Ca_pos)
        loss_N = loss_fn(N_output, N_pos)
        loss_C = loss_fn(C_output, C_pos)
        loss_O = loss_fn(O_output, O_pos)
        loss = loss_weight[0] * loss_Ca + loss_weight[1] * loss_N + \
               loss_weight[2] * loss_C + loss_weight[3] * loss_O

        # print(Ca_output.cpu().numpy().tolist())
        pred_Ca += batch_de_normalize(Ca_output[:].detach().cpu().numpy(), upper_left_corner, lower_right_corner, _offset)
        pred_N  += batch_de_normalize(N_output[:].detach().cpu().numpy(), upper_left_corner, lower_right_corner, _offset)
        pred_C  += batch_de_normalize(C_output[:].detach().cpu().numpy(), upper_left_corner, lower_right_corner, _offset)
        pred_O  += batch_de_normalize(O_output[:].detach().cpu().numpy(), upper_left_corner, lower_right_corner, _offset)

        # print(Ca_output)
        used_time = round(time.time() - curr_time, 4)
        print("[Batch:{}--{}s] \tCa_L1:{}\tN_L1:{}\tC_L1:{}\tO_L1:{}\tloss:{}".format(
            batch_idx, used_time, loss_Ca, loss_N, loss_C, loss_O, loss)
        )
        test_loss += loss.item()
        n_batches += 1

        if n_batches % 5 == 0:      # 5
            break

    test_loss /= n_batches
    writelog(f_l1_loss_log, "time: {}".format(time.asctime(time.localtime(time.time()))))
    writelog(f_l1_loss_log, 'Test loss : ' + str(test_loss))

    plot_coords_diffs(pred_Ca, gt_Ca, target='Ca', _type='bin', save_dir='./imgs/{}/'.format(args.amino_type))
    plot_coords_diffs(pred_N, gt_N, target='N', _type='bin', save_dir='./imgs/{}/'.format(args.amino_type))
    plot_coords_diffs(pred_C, gt_C, target='C', _type='bin', save_dir='./imgs/{}/'.format(args.amino_type))
    plot_coords_diffs(pred_O, gt_O, target='O', _type='bin', save_dir='./imgs/{}/'.format(args.amino_type))

    return test_loss



def writelog(file, line):
    file.write(line + '\n')
    print(line)


if __name__ == '__main__':
    curr_amino_type = args.amino_type.upper()
    if curr_amino_type not in AMINO_ACIDS:
        print("Check amino type, the type {} is invalid".format(curr_amino_type))
        raise ValueError
    checkpoints_dir = './checkpoints/uni_models/{}'.format(curr_amino_type) + 'observation_' + str(datetime.datetime.now().strftime('%H.%M.%S')) + '/'
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    model = My_3D_CNN(in_channels=1, )
    model.to(device)
    # print(model.parameters)

    loss_fn = nn.SmoothL1Loss(beta=0.5)
    optimizer = Adam(list(model.parameters()), lr=LR, weight_decay=w_decay, betas=(0.9, 0.99))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_ratio)

    # train_set = AminoAcidDataset(index_csv='../datas/split/train.csv', )
    # valid_set = AminoAcidDataset(index_csv='../datas/split/valid.csv', )
    # test_set = AminoAcidDataset(index_csv='../datas/split/test.csv', )
    train_set = uniAminoTypeDataset(amino_type=curr_amino_type, index_csv='../datas/split/train.csv', )
    valid_set = uniAminoTypeDataset(amino_type=curr_amino_type, index_csv='../datas/split/valid.csv', )
    test_set  = uniAminoTypeDataset(amino_type=curr_amino_type, index_csv='../datas/split/test.csv', )


    test_set.set_mode(2)
    valid_set.set_mode(2)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)


    min_l1_loss = np.inf

    for epoch in range(n_epochs):
        writelog(f_l1_loss_log, '------ Epoch ' + str(epoch))
        writelog(f_l1_loss_log, 'Training')
        train(epoch, model, train_loader, optimizer, _scheduler=scheduler)

        writelog(f_l1_loss_log, 'Validation')
        valid_loss = test(epoch, "valid", model, valid_loader, optimizer=None, _scheduler=None)

        writelog(f_l1_loss_log, '\nTest Part')
        test_loss = test(epoch, "test", model, test_loader, optimizer=None, _scheduler=None)
        if test_loss < min_l1_loss:
            # ckpnt_dir_path = os.path.join(checkpoints_dir, "/model")
            # if not os.path.exists(ckpnt_dir_path):
            #     os.makedirs(ckpnt_dir_path)
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, "epoch-{}_3D_CNN.pt".format(epoch)))
            writelog(f_l1_loss_log, 'Models at Epoch ' + '/' + str(epoch) + ' are saved!')
            best_epoch = epoch

        scheduler.step()



