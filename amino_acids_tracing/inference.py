"""
    TODO:   1. Using beam-search to generate
            2. Generate type-seq and correspondingly positions seperately

"""
import os
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader
from TraceFormer import Transformer, get_pad_mask, get_subsequent_mask
from amino_fea_loader import AminoFeatureDataset
from Config import Config

cfg = Config()


def cal_performance(pred_seq, pred_pos, gt_seq, gt_pos, trg_pad_idx, 
             seq_weight=1.0, pos_weight=1.0,  
            smoothing=False):
    """
        Return amino-sequence-type loss and position loss 
        Apply label smoothing if needed.
    """
    # print("In cal performance:", gt_pos.shape)
    bs, seq_len, dim = gt_pos.shape
    non_pad_mask = gt_seq.ne(0)
    # print("Non-Zero mask shape: ", non_pad_mask.shape)
    _pred_seq = pred_seq.view(-1, pred_seq.size(-1))
    _gt_seq = gt_seq.view(-1).long()
    _pred_pos = pred_pos.view(-1, pred_pos.size(-1))
    _gt_pos = gt_pos.view(-1, gt_pos.size(-1))

    amino_seq_loss = cal_seq_loss(_pred_seq, _gt_seq, trg_pad_idx)
    # print("seq loss:\n", amino_seq_loss)
    amino_pos_loss = cal_pos_loss(_pred_pos, _gt_pos, non_pad_mask=non_pad_mask.view(-1))               # TODO: if need this index？
    # loss = amino_seq_loss * seq_weight +  amino_pos_loss * pos_weight
    # loss /= (seq_weight + pos_weight)
    # print("Losses are:{} \t {}".format(amino_pos_loss.item(), amino_seq_loss.item()))


    pred = _pred_seq.max(1)[1]
    gt_type = _gt_seq.contiguous().view(-1)
    non_pad_mask = gt_seq.ne(0).view(-1)
    n_correct = pred.eq(gt_type).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    # print("In cal-performance: ", pred.shape)
    non_pad_index = torch.arange(len(non_pad_mask))[non_pad_mask]

    check_nums = 20
    rand_idx = torch.randint(len(non_pad_index), (check_nums, ))

    return amino_seq_loss, amino_pos_loss, n_correct, n_word



def cal_seq_loss(pred_amino_seq, gt_seq, trg_pad_idx):
    clf_loss = F.cross_entropy(pred_amino_seq, gt_seq, ignore_index=0, reduction="sum").to(device)
    return clf_loss


def cal_pos_loss(pred_amino_pos, gt_pos, non_pad_mask=None):
    smooth_l1_fn = nn.SmoothL1Loss(reduction='sum', beta=0.1)                      # beta = 1.0, in original version
    pos_loss = smooth_l1_fn(pred_amino_pos[non_pad_mask], gt_pos[non_pad_mask])
    
    n_aminos = non_pad_mask.sum().item()
    return pos_loss


def _inference_test(cfg, device):
    model = Transformer(23, 23)
    model = nn.DataParallel(model)

    ckpnt = torch.load(cfg.best_model_path)
    _state_dict = ckpnt['model']
    model.load_state_dict(_state_dict)
    
    print(model.parameters)

    test_data = AminoFeatureDataset(index_csv='../datas/tracing_data2/test.csv',  z_score_coords=False)
    test_loader = DataLoader(test_data, shuffle=False)

    for batch_idx, batch_data in enumerate(test_loader):
        seq_data_array = batch_data[0].to(torch.float32).to(device)   
        labels = batch_data[1].to(torch.float32).to(device)           # batch x seq_len(512) x 13

        pred_seq, pred_pos = model(seq_data_array, labels)
        break


def load_model():
    pass


def function_test():
    a = torch.tensor([[1, 2, 3], [4, 5, 6]]).float()
    b = torch.tensor([[4, 1, 3], [8, 3, 1]]).float()

    l1_fn_mean = nn.L1Loss(reduction='mean')        # mean is element wise, but not object wise
    l1_fn_sum = nn.L1Loss(reduction='sum')
    loss_l1_mean = l1_fn_mean(a, b)
    loss_l1_sum  = l1_fn_sum(a , b)
    print("L1 loss result:")
    print("Mean: {}\t Sum: {}".format(loss_l1_mean, loss_l1_sum))

    l2_fn_mean = nn.MSELoss(reduction='mean')
    l2_fn_sum = nn.MSELoss(reduction='sum')
    loss_l2_mean = l2_fn_mean(a, b)
    loss_l2_sum  = l2_fn_sum(a , b)
    print("L2 loss result:")
    print("Mean: {}\t Sum: {}".format(loss_l2_mean, loss_l2_sum))



if __name__ == '__main__':
    # function_test()
    print("GPU available: {}".format(torch.cuda.device_count()))
    gpu_id = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print("GPU available: {}".format(torch.cuda.device_count()))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_ids = [0, 1]
    _inference_test(cfg, device)