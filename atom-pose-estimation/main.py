#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py    
@Contact :   roserland@gmail.com
Try to use 3D U-Net to regress some 'Key Points' among an amino acid cube, such as:
    1. The Ca atom, which will be used in Protein Backbone Tracing
    2. The N atom in NH2, will be used to calculate Ca-N-CO angle, to determine amino acid cube's spatial position.
    3. The O atom in CO,
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/9 10:25 下午   fanzw      1.0         None
'''

"""
This file is a elementary test, we will:
    1. First, use 3D-UNet to regress a single atom coordinates: [x, y, z]
    2. Second, use some separate model to predict their position among such as 'Ca, N, C, O' atoms 
    3. Finally, using a single model to predict these 3 or 4 atoms' position, for there must be some spatial connection
       between these atoms.
"""

# import lib
from light_3DUNet import My_3D_CNN
from cube_loader import *
from torch.optim import Adam, lr_scheduler
import os, time, datetime
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np

# prepare environment
gpu_id = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
print('gpu ID is ', str(gpu_id))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Current Device:", device)

LR = 0.01
w_decay = 0.001
lr_decay_step_size = 10
lr_decay_ratio = 0.2
batch_size = 64
n_epochs = 120
loss_weight = np.array([1.0, 1.0, 1.0, 1.0]) / 4

log_dir = './log_files/'
f_l1_loss_log = open(os.path.join(log_dir, 'key_point_pos_estimation_loss.txt'),  'a')


def train(epoch, model, train_loader, optimizer, _scheduler):
    model.train()
    train_loss = 0
    n_batches = 0
    for batch_idx, data in enumerate(train_loader):
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
        print("[Batch:{}] \tCa_L1:{}\tN_L1:{}\tC_L1:{}\tO_L1:{}\tloss:{}".format(
            batch_idx, loss_Ca, loss_N, loss_C, loss_O, loss)
        )

        train_loss += loss.item()
        n_batches += 1
        if n_batches % 100 == 0:
            break

    train_loss = train_loss / n_batches
    print("Epoch {} Train loss: {}".format(epoch, train_loss))
    writelog(f_l1_loss_log, "Train loss: {}".format(train_loss))


def test(epoch, phase, model, test_loader, optimizer, _scheduler):
    model.eval()

    test_loss = 0.0
    n_batches = 0.0
    print("[Test/Valid] -- [Epoch: {}]".format(epoch))

    for batch_idx, data in enumerate(test_loader):
        cube_data_array = data[0].to(torch.float32).to(device)
        Ca_pos = data[1].to(torch.float32).to(device)
        N_pos = data[2].to(torch.float32).to(device)
        C_pos = data[3].to(torch.float32).to(device)
        O_pos = data[4].to(torch.float32).to(device)
        # print(C_pos)

        Ca_output, N_output, C_output, O_output = model(cube_data_array)
        loss_Ca = loss_fn(Ca_output, Ca_pos)
        loss_N = loss_fn(N_output, N_pos)
        loss_C = loss_fn(C_output, C_pos)
        loss_O = loss_fn(O_output, O_pos)
        loss = loss_weight[0] * loss_Ca + loss_weight[1] * loss_N + \
               loss_weight[2] * loss_C + loss_weight[3] * loss_O

        # print(Ca_output)
        print("[Batch:{}] \tCa_L1:{}\tN_L1:{}\tC_L1:{}\tO_L1:{}\tloss:{}".format(
            batch_idx, loss_Ca, loss_N, loss_C, loss_O, loss)
        )
        test_loss += loss.item()
        n_batches += 1

        if n_batches % 25 == 0:
            break

    test_loss /= n_batches
    writelog(f_l1_loss_log, "time: {}".format(time.asctime(time.localtime(time.time()))))
    writelog(f_l1_loss_log, 'Test loss : ' + str(test_loss))

    return test_loss



def writelog(file, line):
    file.write(line + '\n')
    print(line)


checkpoints_dir = './checkpoints/' + 'observation_' + str(datetime.datetime.now().strftime('%H.%M.%S')) + '/'
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

if __name__ == '__main__':

    model = My_3D_CNN(in_channels=1, )
    # print(model.parameters)

    loss_fn = nn.L1Loss()
    optimizer = Adam(list(model.parameters()), lr=LR, weight_decay=w_decay, betas=(0.9, 0.99))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_ratio)

    train_set = AminoAcidDataset(index_csv='./datas/split/train.csv', )
    valid_set = AminoAcidDataset(index_csv='./datas/split/valid.csv', )
    test_set = AminoAcidDataset(index_csv='./datas/split/test.csv', )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    min_l1_loss = np.inf

    # *****************************************************************
    # # Test data:
    # for batch_idx, data in enumerate(train_loader):
    #     cube_data_array = data[0].to(torch.float32).to(device)
    #     Ca_pos = data[1].to(torch.float32).to(device)
    #     N_pos = data[2].to(torch.float32).to(device)
    #     C_pos = data[3].to(torch.float32).to(device)
    #     O_pos = data[4].to(torch.float32).to(device)
    #     print(Ca_pos[0], N_pos[0], C_pos[0], O_pos[0])
    #
    # for batch_idx, data in enumerate(test_loader):
    #     cube_data_array = data[0].to(torch.float32).to(device)
    #     Ca_pos = data[1].to(torch.float32).to(device)
    #     N_pos = data[2].to(torch.float32).to(device)
    #     C_pos = data[3].to(torch.float32).to(device)
    #     O_pos = data[4].to(torch.float32).to(device)
    #     print(Ca_pos[0], N_pos[0], C_pos[0], O_pos[0])
    #
    # for batch_idx, data in enumerate(valid_loader):
    #     cube_data_array = data[0].to(torch.float32).to(device)
    #     Ca_pos = data[1].to(torch.float32).to(device)
    #     N_pos = data[2].to(torch.float32).to(device)
    #     C_pos = data[3].to(torch.float32).to(device)
    #     O_pos = data[4].to(torch.float32).to(device)
    #     print(Ca_pos[0], N_pos[0], C_pos[0], O_pos[0])

    # ****************************************************

    for epoch in range(n_epochs):
        writelog(f_l1_loss_log, '------ Epoch ' + str(epoch))
        writelog(f_l1_loss_log, 'Training')
        train(epoch, model, train_loader, optimizer, _scheduler=scheduler)

        writelog(f_l1_loss_log, 'Validation')
        valid_loss = test(epoch, "valid", model, valid_loader, optimizer=None, _scheduler=None)

        writelog(f_l1_loss_log, '\nTest Part')
        test_loss = test(epoch, "test", model, test_loader, optimizer=None, _scheduler=None)
        if test_loss < min_l1_loss:
            # torch.save(model, checkpoints_dir + 'model/' + '/' + str(epoch) + '_3d_cnn.pt')
            writelog(f_l1_loss_log, 'Models at Epoch ' + '/' + str(epoch) + ' are saved!')
            best_epoch = epoch

        scheduler.step()



