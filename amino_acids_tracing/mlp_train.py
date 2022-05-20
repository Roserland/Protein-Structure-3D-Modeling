import argparse
import os, time, datetime, math, random
from turtle import forward
from wsgiref.simple_server import demo_app
from cv2 import log

from numpy import gradient
from TraceFormer import ScheduledOptim
from my_datasets import MLP_Dataset
from simple_models import MLP
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.autograd import gradcheck
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Config import Config
import argparse

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn import metrics


print("GPU available: {}".format(torch.cuda.device_count()))
gpu_id = "0, 1"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
print("GPU available: {}".format(torch.cuda.device_count()))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1]

cfg = Config()
parser = argparse.ArgumentParser(description="Amino-acids Affinity Training")
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--lr_mul", type=float, default=0.0001)
parser.add_argument("--n_warmup_epochs", type=int, default=2000)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--focal_alpha", type=float, default=0.5)
parser.add_argument("--focal_gamma", type=float, default=2)
parser.add_argument("--using_gt", action="store_true", default=False)
parser.add_argument("--warm_up", action="store_true", default=False)
args = parser.parse_args()

cfg = Config()
cfg.lr = args.lr
cfg.lr_mul = args.lr_mul
cfg.n_warmup_steps = args.n_warmup_epochs
cfg.bacth_size = args.batch_size
cfg.alpha = args.focal_alpha
cfg.gamma = args.focal_gamma
cfg.using_gt = args.using_gt
cfg.warmup = args.warm_up

def cal_performance(pred_linkage, gt_linkage, ignore_index=-1, focal_loss=True, 
                    seq_weight=1.0, pos_weight=1.0,  
                    smoothing=False):
    seq_len, dim = pred_linkage.shape
    non_pad_mask = gt_linkage.ne(ignore_index)

    _pred = pred_linkage[non_pad_mask].contiguous().view(-1)
    _gt = gt_linkage[non_pad_mask].contiguous().view(-1)

    # print("pred len: {}, GT len: {}".format(_pred.shape, _gt.shape))

    focal_bce_loss = BCE_FocalLoss_withLogits(gamma=cfg.gamma, alpha=cfg.alpha, reduction='mean')
    loss = focal_bce_loss(_pred, _gt)

    pred_linking = (torch.sigmoid(_pred) > 0.5).long()
    tn, fp, fn, tp = confusion_matrix(_gt.cpu(), pred_linking.cpu()).ravel()
    total = tn + fp + fn + tp

    return _pred, _gt, loss, tn, fp, fn, tp, total


def multiclass_cal_performance(pred_linkage, gt_linkage, ignore_index=-1, focal_loss=True, 
                                seq_weight=1.0, pos_weight=1.0,  
                                smoothing=False):
    seq_len, dim = pred_linkage.shape
    non_pad_mask = gt_linkage.ne(ignore_index)

    _pred = pred_linkage[non_pad_mask].contiguous().view(-1)
    _gt = gt_linkage[non_pad_mask].contiguous().view(-1)

    # print("pred len: {}, GT len: {}".format(_pred.shape, _gt.shape))

    focal_loss = MultiClass_FocalLoss(gamma=cfg.gamma, alpha=cfg.alpha, reduction='mean')
    loss = focal_loss(_pred, _gt)

    pred_linking = F.log_softmax(pred_linking).argmax(dim=1)
    conf_max = confusion_matrix(_gt.cpu(), pred_linking.cpu())

    # total = tn + fp + fn + tp

    return _pred, _gt, loss, pred_linking


class BCE_FocalLoss_withLogits(nn.Module):
    def __init__(self, gamma=2, alpha=0.5, reduction='mean') -> None:
        super(BCE_FocalLoss_withLogits, self).__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV1()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

class MultiClass_FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.5, size_average=True) -> None:
        super(MultiClass_FocalLoss).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001
    
    def forward(self, logits, labels):
        """
        calculate loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)   # N x label length
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()

        assert logits.size(0) == labels.size(0)
        assert logits.size(2) == labels.size(1)

        batch_size = logits.size(0)
        labels_length = logits.size(1)
        seq_length = logits.size(2)

        # transpose labels into labels one-hot
        new_label = labels.unsequeeze(1)
        label_onehot = torch.zeros([batch_size, labels_length, seq_length]).scatter_(1, new_label, 1)

        # cal log
        log_p = F.log_softmax(logits)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        focal_loss = -self.alpha * (sub_pt) ** self.gamma * log_p
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


def train_epoch(mlp_model, training_data, optimizer, opt, device, smoothing):
    ''' Epoch operation in training phase'''

    mlp_model.train()
    total_loss, link_acc, non_linl_acc = 0, 0, 0
    tp_all, fp_all, fn_all, tn_all = 0, 0, 0, 0

    all_pred_scores = np.array([])
    all_gts = np.array([])

    desc = '  - (Training)   '
    print_grad = True
    for batch_data in tqdm(training_data, mininterval=2, desc=desc, leave=False):

        # prepare data
        seq_data_array = batch_data[0].to(torch.float32).cuda(device=device_ids[0])     # batch_size * 4096 * 24
        labels = batch_data[1].to(torch.float32).cuda(device=device_ids[0])             # batch x seq_len(512) x 13
                                                                                        # Need to add a BOS token
        seq_data_array = torch.cat((seq_data_array[:, :, :3], seq_data_array[:, :, 12:15]), dim=2)
        seq_data_array = seq_data_array.reshape(-1, 6)

        # seq_data_array = seq_data_array.reshape(-1, 24)
        labels = labels.reshape(-1)

        # forward
        optimizer.zero_grad()
        pred_linkage = mlp_model(seq_data_array)

        # backward and update parameters
        # loss, link_num, non_link_num, len(gt_linking), len(gt_non_linking)
        _pred, _gt, link_loss, tn, fp, fn, tp, total = \
            cal_performance(pred_linkage, labels, ignore_index=-1, focal_loss=True)

        loss = link_loss
        loss.backward()

        # check gradient ---- print
        if print_grad:
            print_grad = False
            for name, params in mlp_model.named_parameters():
                print("\n--> name: ", name)
                print("    grad_value mean: {}\t std: {}".format(torch.mean(params.grad), torch.std(params.grad)))
        
        optimizer.step_and_update_lr()

        # note keeping
        all_pred_scores = np.concatenate((all_pred_scores, _pred.cpu().detach().numpy()))
        all_gts = np.concatenate((all_gts, _gt.cpu().detach().numpy()))
        # all_gts += _gt.cpu().detach().numpy().tolist()
        
        total_loss += loss.item()
        tp_all += tp
        tn_all += tn
        fp_all += fp
        fn_all += fn
    
    fpr, tpr, thres = metrics.roc_curve(all_gts, all_pred_scores)
    auc_score = metrics.auc(fpr, tpr)
    _precision, _recall, thresholds = precision_recall_curve(all_gts, all_pred_scores)
    # _PR_AUC = metrics.auc(_precision, _recall)
    print("   Train ---> AUC: {}".format(auc_score))
    # print("   Train ---> AUC: {}\t PR-AUC: {}".format(auc_score, _PR_AUC))

    total = tp_all + tn_all + fp_all +fn_all
    acc = (tp_all + tn_all) / total
    precision_all = tp_all / (tp_all + fp_all)
    recall_all = tp_all / (tp_all + fn_all)
    return total_loss, tp_all, fn_all, fp_all, tn_all, total, acc, precision_all, recall_all, auc_score



def eval_epoch(mlp_model, valid_data,  device=device, phase="Validation"):
    ''' Epoch operation in training phase'''

    mlp_model.eval()
    total_loss, link_acc, non_linl_acc = 0, 0, 0
    tp_all, fp_all, fn_all, tn_all = 0, 0, 0, 0

    all_pred_scores = np.array([])
    all_gts = np.array([])

    desc = '  - (Validing)   '
    with torch.no_grad():
        for batch_data in tqdm(valid_data, mininterval=2, desc=desc, leave=False):

            # prepare data
            seq_data_array = batch_data[0].to(torch.float32).cuda(device=device_ids[0])     #.to(device)
            labels = batch_data[1].to(torch.float32).cuda(device=device_ids[0])             # batch x seq_len(512) x 13

            seq_data_array = torch.cat((seq_data_array[:, :, :3], seq_data_array[:, :, 12:15]), dim=2)
            seq_data_array = seq_data_array.reshape(-1, 6)
            
            # seq_data_array = seq_data_array.reshape(-1, 24)
            labels = labels.reshape(-1)

            # forward
            pred_linkage = mlp_model(seq_data_array)

            # backward and update parameters
            # loss, link_num, non_link_num, len(gt_linking), len(gt_non_linking)
            _pred, _gt, link_loss, tn, fp, fn, tp, total = \
                cal_performance(pred_linkage, labels, ignore_index=-1, focal_loss=True)

            loss = link_loss

            # note keeping
            total_loss += loss.item()
            tp_all += tp
            tn_all += tn
            fp_all += fp
            fn_all += fn

            all_pred_scores = np.concatenate((all_pred_scores, _pred.cpu().detach().numpy()))
            all_gts = np.concatenate((all_gts, _gt.cpu().detach().numpy()))

        fpr, tpr, thres = metrics.roc_curve(all_gts, all_pred_scores)
        auc_score = metrics.auc(fpr, tpr)
        _precision, _recall, thresholds = precision_recall_curve(all_gts, all_pred_scores)
        # _PR_AUC = metrics.auc(_precision, _recall)
        print("   Valid ---> AUC: {}".format(auc_score))
        # print("   Valid ---> AUC: {}\t PR-AUC: {}".format(auc_score, _PR_AUC))

        total = tp_all + tn_all + fp_all +fn_all
        acc = (tp_all + tn_all) / total
        precision_all = tp_all / (tp_all + fp_all)
        recall_all = tp_all / (tp_all + fn_all)
        return total_loss, tp_all, fn_all, fp_all, tn_all, total, acc, precision_all, recall_all, auc_score


def train(model, training_data, validation_data, test_data, optimizer, cfg, smoothing, device=device):
    # Use tensorboard to plot curves, e.g. perplexity, accuracy, learning rate
    if cfg.use_tb:
        print("[Info] Use Tensorboard")
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(cfg.output_dir, 'tensorboard'))
    
    time_str = time.strftime("%Y-%m-%d_%H_%M", time.localtime())
    # os.makedirs(os.path.join(cfg.logs_dir, time_str), True)
    log_train_file = os.path.join(cfg.logs_dir, time_str + '_train.log')
    log_valid_file = os.path.join(cfg.logs_dir, time_str + '_valid.log')

    print('[Info] Training performance will be written to file: {} and {}'.format(
        log_train_file, log_valid_file))
    
    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,accuracy,auc\n')
        log_vf.write('epoch,loss,accuracy,auc\n')
    
    def print_performances(header, loss, tp, fn, fp, tn, total, acc, precision, recall, start_time, lr):
        print('  - {header:12} loss: {loss: 8.5f}, tp:{tp:9}, fn: {fn:9}, fp:{fp:9}, tn: {tn:9}, total: {total:12}, \
                acc:{acc:8.5f}%, precision: {precision: 8.5f}%, recall: {recall: 8.5f}%\
                lr: {lr:8.5f}, '\
                'elapse: {elapse:3.3f} min'.format(
                  header=f"({header})", loss=loss, tp=tp, fn=fn, fp=fp, tn=tn, total=total,
                  acc=acc*100, precision=precision*100, recall=recall*100,
                  elapse=(time.time()-start_time)/60, lr=lr))
    
    valid_losses = []
    for epoch_i in range(cfg.num_epochs):  
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_tp, train_fn, train_fp, train_tn, train_total, train_acc, train_precision, train_recall, train_auc = \
                train_epoch(model, training_data, optimizer, cfg, device, smoothing=cfg.label_smoothing)

        lr = optimizer._optimizer.param_groups[0]['lr']
        print_performances('Training',  train_loss, train_tp, train_fn, train_fp, train_tn, train_total,
                                        train_acc, train_precision, train_recall, 
                                        start, lr)

        start = time.time()
        valid_loss, valid_tp, valid_fn, valid_fp, valid_tn, valid_total, valid_acc, valid_precision, valid_recall, valid_auc = \
                eval_epoch(model, validation_data, device, phase="Validation")
        print_performances('Validation', valid_loss, valid_tp, valid_fn, valid_fp, valid_tn, valid_total,
                                         valid_acc, valid_precision, valid_recall, 
                                         start, lr)

        valid_losses += [valid_loss]

        checkpoint = {'epoch': epoch_i, 'settings': cfg, 'model': model.state_dict()}

        if cfg.save_mode == 'all':
            model_name = 'model_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_acc)
            torch.save(checkpoint, model_name)
            if valid_acc == 1.0:
                best_model_name = "model_accu_100.000_pos_los{}.chkpt".format(valid_loss)
                torch.save(checkpoint, best_model_name)
        elif cfg.save_mode == 'best':
            model_name = 'model.chkpt'
            if valid_loss <= min(valid_losses):
                torch.save(checkpoint, os.path.join(cfg.output_dir, model_name))
                print('    - [Info] The checkpoint file has been updated.')
        
        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss: 8.5f},{accu:3.3f},{auc:8.8f}\n'.format(
                epoch=epoch_i, loss=train_loss,
                accu=100*train_acc, auc=train_auc))
            log_vf.write('{epoch},{loss: 8.5f},{accu:3.3f},{auc:8.8f}\n'.format(
                epoch=epoch_i, loss=valid_loss,
                accu=100*train_acc, auc=valid_auc))

        if cfg.use_tb:
            tb_writer.add_scalars('ppl', {'train': train_loss, 'val': valid_loss}, epoch_i)
            tb_writer.add_scalars('accuracy', {'train': train_acc*100, 'val': train_acc*100}, epoch_i)
            tb_writer.add_scalar('learning_rate', lr, epoch_i)
        
        # if ((epoch_i+1) % 5 == 1):
        #     start = time.time()
        #     test_valid(model, protein_id='6Q29')


def test_valid(model, protein_id, max_len=512, pad=0, shuffle=False):
    model.eval()
    fea_src_dir='/mnt/data/zxy/relat_coors_stage3-amino-keypoint-vectors/'
    amino_acids_dir = os.path.join(fea_src_dir, protein_id)
    amino_file_list = os.listdir(amino_acids_dir)

    if shuffle:
        random.shuffle(amino_file_list)        

    if len(amino_file_list) > max_len:
        print("The detected amino acids num is larger than {}. Split the set or change the 'max_len' ".format(max_len))

    data_array = np.zeros((max_len, 13)) + pad
    amino_index_list = []
    for i, amino_file in enumerate(amino_file_list):
        data_array[i] = np.load(os.path.join(amino_acids_dir, amino_file)).reshape(-1)
        amino_index_list.append(int(amino_file[:-4].split('_')[1]))

    input_data = torch.from_numpy(data_array).to(torch.float32).unsqueeze(0) # .cuda(device=device_ids[0])
    input_data = input_data.reshape(-1, 24)
    # input_ata.cuda(device=self.device_ids[0])

    print(amino_index_list)
    with torch.no_grad():
        pred_linkage = model(input_data)[0]
        print("Original Output of the Model is :\n", pred_linkage)
        sigmoid_pred = torch.sigmoid(pred_linkage)
    print(sigmoid_pred)

    order = np.argsort(amino_index_list)
    print(order.shape)
    print("Order:\n", order)
    pred_res = sigmoid_pred[:len(order), :len(order)].cpu().numpy()
    binary_res = (pred_res >= 0.5)
    print("Binary res: \n", binary_res.shape)
    print("Binary res: \n", binary_res)
    sorted_binary = binary_res[order][:, order]
    print(sorted_binary)
    sorted_sigmoid = pred_res[order][:, order]

    print(sorted_sigmoid)
    print("The true linkages are:")
    value_list = []
    for i in range(1, len(order)):
        value_list.append(sorted_sigmoid[i-1, i])
    print(np.array(value_list))
    bool_list = []
    for i in range(1, len(order)):
        bool_list.append(sorted_binary[i-1, i])
    print(np.array(bool_list))
    return sigmoid_pred


def main():
    # transformer = Transformer(22, 22).to(device)
    linkage_model = MLP(input_dim=6)
    linkage_model = torch.nn.DataParallel(linkage_model, device_ids=device_ids)
    linkage_model = linkage_model.cuda(device=device_ids[0]) 

    if cfg.warmup:
        optimizer = ScheduledOptim(
            optim.Adam(linkage_model.parameters(), betas=(0.9, 0.98), eps=1e-09),
            cfg.lr_mul, cfg.d_model, cfg.n_warmup_steps)
    else:
        print("**** Using fixed optimiizer ****")
        # optimizer = ScheduledOptim(
        #     optim.Adam(linkage_model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        #     cfg.lr, cfg.d_model, cfg.n_warmup_steps, cfg.warmup)
        optimizer = ScheduledOptim(
            optim.SGD(linkage_model.parameters(), lr=cfg.lr, momentum=0.9, dampening=0.5, weight_decay=0.01, nesterov=False),
            cfg.lr, cfg.d_model, cfg.n_warmup_steps, cfg.warmup)
    
    train_set = MLP_Dataset(index_csv='../datas/tracing_data2/train.csv', multi_class=False, using_gt=cfg.using_gt, shuffle=True, coords_norm=True,)
    valid_set = MLP_Dataset(index_csv='../datas/tracing_data2/valid.csv', multi_class=False, using_gt=cfg.using_gt, shuffle=False, coords_norm=True,)
    test_set  = MLP_Dataset(index_csv='../datas/tracing_data2/test.csv',  multi_class=False, using_gt=cfg.using_gt, shuffle=False, coords_norm=True,)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=cfg.bacth_size * len(device_ids), num_workers=4)
    valid_loader = DataLoader(valid_set, shuffle=True, batch_size=cfg.bacth_size * len(device_ids), num_workers=4)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=cfg.bacth_size * len(device_ids), num_workers=4)
    train(linkage_model, train_loader, valid_loader, test_loader, optimizer, cfg, smoothing=False, device=device)
    


if __name__ == "__main__":
    # model = Transformer(n_src_vocab=22,  n_trg_vocab=22)
    main()