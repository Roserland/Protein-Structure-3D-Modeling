#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py    
@Contact :   roserland@gmail.com
Try to use Hourglass3D to regress some 'Key Points' among an amino acid cube, such as:
    1. The Ca atom, which will be used in Protein Backbone Tracing
    2. The N atom in NH2, will be used to calculate Ca-N-CO angle, to determine amino acid cube's spatial position.
    3. The O atom in CO,
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/9 10:25 下午   fanzw      1.0         None
'''


# import lib
# from light_3DUNet import My_3D_CNN
from atom_pose_estimation.layers import HourGlass3DNet, HourGlass3DNet_2
from atom_pose_estimation.cube_loader import *
from atom_pred_visualize import plot_coords_diffs
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
batch_size = 256
n_epochs = 120
loss_weight = [10, 1.0, 10, 10]

log_dir = './log_files/hourglass/'
f_l1_loss_log = open(os.path.join(log_dir, 'key_point_pos_estimation_loss.txt'),  'a')


def train(epoch, model, train_loader, optimizer, _scheduler):
    model.train()
    train_loss = 0
    n_batches = 0
    for batch_idx, data in enumerate(train_loader):
        curr_time = time.time()
        cube_data_array = data[0].to(torch.float32).to(device)
        heatmap_gt = data[1].to(torch.float32).to(device)           # Batch x 4 x 16 x 16 x16
        # print(C_pos)        
        optimizer.zero_grad()

        heatmap_pred = model(cube_data_array)
        # torch.transpose ?
        loss_Ca = loss_fn(heatmap_pred[:, 0], heatmap_gt[:, 0])
        loss_N = loss_fn(heatmap_pred[:, 1], heatmap_gt[:, 1])
        loss_C = loss_fn(heatmap_pred[:, 2], heatmap_gt[:, 2])
        loss_O = loss_fn(heatmap_pred[:, 3], heatmap_gt[:, 3])

        
        loss = loss_weight[0] * loss_Ca + loss_weight[1] * loss_N + \
               loss_weight[2] * loss_C + loss_weight[3] * loss_O

        loss.backward()
        optimizer.step()
        # print(Ca_output)
        used_time = round(time.time() - curr_time, 4)
        print("[Batch:{}--{}ms] \tCa_L1:{}\tN_L1:{}\tC_L1:{}\tO_L1:{}\tloss:{}".format(
            batch_idx, used_time, loss_Ca, loss_N, loss_C, loss_O, loss)
        )

        train_loss += loss.item()
        n_batches += 1
        if n_batches % 100 == 0:
            break

    train_loss = train_loss / n_batches
    print("Epoch {} Train loss: {}".format(epoch, train_loss))
    writelog(f_l1_loss_log, "Train loss: {}".format(train_loss))


@torch.no_grad()
def test(epoch, phase, model, test_loader, optimizer, _scheduler):
    model.eval()
    test_loss = 0.0
    n_batches = 0.0
    print("[Test/Valid] -- [Epoch: {}]".format(epoch))

    gt_Ca = []; pred_Ca = []
    gt_N  = []; pred_N = []
    gt_C = []; pred_C = []
    gt_O = []; pred_O = []
    relate_gt_Ca_list = []; 
    relate_gt_N_list = []; 
    relate_gt_C_list= []; 
    relate_gt_O_list= []; 

    heatmap_generator = HeatmapGenerator3D(sigma=1.0)
    for batch_idx, data in enumerate(test_loader):
        curr_time = time.time()
        cube_data_array = data[0].to(torch.float32).to(device)
        heatmap_gt = data[1].to(torch.float32).to(device)           # Batch x 4 x 16 x 16 x16

        # print(C_pos)
        relate_gt_Ca_list += data[2][:10].numpy().tolist() # .to(torch.float32).to(device)
        relate_gt_N_list  += data[3][:10].numpy().tolist() # .to(torch.float32).to(device)
        relate_gt_C_list  += data[4][:10].numpy().tolist() # .to(torch.float32).to(device)
        relate_gt_O_list  += data[5][:10].numpy().tolist() # .to(torch.float32).to(device)

        gt_Ca += data[6][:10].numpy().tolist() # .to(torch.float32).to(device)
        gt_N  += data[7][:10].numpy().tolist() # .to(torch.float32).to(device)
        gt_C  += data[8][:10].numpy().tolist() # .to(torch.float32).to(device)
        gt_O  += data[9][:10].numpy().tolist() # .to(torch.float32).to(device)
        # de_normalize
        upper_left_corner = data[10][:10].numpy()
        lower_right_corner = data[11][:10].numpy()
        _offset = data[12][:10].numpy()

        heatmap_pred = model(cube_data_array)
        loss_Ca = loss_fn(heatmap_pred[:, 0], heatmap_gt[:, 0])
        loss_N = loss_fn(heatmap_pred[:, 1], heatmap_gt[:, 1])
        loss_C = loss_fn(heatmap_pred[:, 2], heatmap_gt[:, 2])
        loss_O = loss_fn(heatmap_pred[:, 3], heatmap_gt[:, 3])

        print("Ca L1 loss: ", torch.abs(heatmap_pred[:, 0] - heatmap_gt[:, 0]).sum())
        print("N L1 loss: ", torch.abs(heatmap_pred[:, 1] - heatmap_gt[:, 1]).sum())
        print("C L1 loss: ", torch.abs(heatmap_pred[:, 2] - heatmap_gt[:, 2]).sum())
        print("O L1 loss: ", torch.abs(heatmap_pred[:, 3] - heatmap_gt[:, 3]).sum())

        loss = loss_weight[0] * loss_Ca + loss_weight[1] * loss_N + \
               loss_weight[2] * loss_C + loss_weight[3] * loss_O
        
        Ca_output = heatmap_generator.get_maps_pos(heatmap_pred[:, 0].detach().cpu().numpy())
        N_output = heatmap_generator.get_maps_pos(heatmap_pred[:, 1].detach().cpu().numpy())
        C_output = heatmap_generator.get_maps_pos(heatmap_pred[:, 2].detach().cpu().numpy())
        O_output = heatmap_generator.get_maps_pos(heatmap_pred[:, 3].detach().cpu().numpy())

        print("1: From heatmap:\n", Ca_output[:10].tolist())
        print("*******************************")
        print(relate_gt_Ca_list[:10])
        # print(Ca_output.cpu().numpy().tolist())
        pred_Ca += batch_de_normalize(Ca_output[:10], upper_left_corner, lower_right_corner, _offset)
        pred_N  += batch_de_normalize(N_output[:10], upper_left_corner, lower_right_corner, _offset)
        pred_C  += batch_de_normalize(C_output[:10], upper_left_corner, lower_right_corner, _offset)
        pred_O  += batch_de_normalize(O_output[:10], upper_left_corner, lower_right_corner, _offset)

        # print("2:  From heatmap:\n:", pred_Ca[:10])
        # print("*******************************")
        # print(gt_Ca[:10])

        # print(Ca_output)
        used_time = round(time.time() - curr_time, 4)
        print("[Batch:{}--{}ms] \tCa_L1:{}\tN_L1:{}\tC_L1:{}\tO_L1:{}\tloss:{}".format(
            batch_idx, used_time, loss_Ca, loss_N, loss_C, loss_O, loss)
        )
        test_loss += loss.item()
        n_batches += 1

        if n_batches % 1 == 0:
            break

    test_loss /= n_batches
    writelog(f_l1_loss_log, "time: {}".format(time.asctime(time.localtime(time.time()))))
    writelog(f_l1_loss_log, 'Test loss : ' + str(test_loss))

    plot_coords_diffs(pred_Ca, gt_Ca, target='Ca', _type='bin', save_dir='./imgs/{}/'.format("from_heatmap"))
    plot_coords_diffs(pred_N, gt_N, target='N', _type='bin', save_dir='./imgs/{}/'.format("from_heatmap"))
    plot_coords_diffs(pred_C, gt_C, target='C', _type='bin', save_dir='./imgs/{}/'.format("from_heatmap"))
    plot_coords_diffs(pred_O, gt_O, target='O', _type='bin', save_dir='./imgs/{}/'.format("from_heatmap"))

    return test_loss



def writelog(file, line):
    file.write(line + '\n')
    print(line)


checkpoints_dir = './hourglass_checkpoints/' + 'observation_' + str(datetime.datetime.now().strftime('%H.%M.%S')) + '/'
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

if __name__ == '__main__':

    model = HourGlass3DNet(in_channels=1, out_channels=4)
    # model = HourGlass3DNet_2(in_channels=1, out_channels=4, regression=True)
    model.to(device)
    # print(model.parameters)

    loss_fn = nn.MSELoss(reduction='mean')           # if 'mean', the loss is so small
    optimizer = Adam(list(model.parameters()), lr=LR, weight_decay=w_decay, betas=(0.9, 0.99))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_ratio)

    train_set = AminoAcidDataset(index_csv='../datas/split/train.csv', standard_size=[16, 16, 16], gt_type="heatmap")
    valid_set = AminoAcidDataset(index_csv='../datas/split/valid.csv', standard_size=[16, 16, 16], gt_type="heatmap")
    test_set = AminoAcidDataset(index_csv='../datas/split/test.csv',   standard_size=[16, 16, 16], gt_type="heatmap")

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



