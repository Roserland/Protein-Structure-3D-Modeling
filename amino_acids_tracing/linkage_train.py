import argparse
import os, time, math
from turtle import forward

from numpy import gradient
from TraceFormer import Transformer, get_pad_mask, get_subsequent_mask, ScheduledOptim, Encoder, Decoder, LinkageFormer
from amino_fea_loader import AminoFeatureDataset, LinkageSet
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.autograd import gradcheck
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Config import Config

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn import metrics


print("GPU available: {}".format(torch.cuda.device_count()))
gpu_id = "0, 1"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
print("GPU available: {}".format(torch.cuda.device_count()))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1]


seq_clf_loss_weight = 1.0
seq_pos_loss_weight = 1.0

cfg = Config()

# TODO: 1. BCE loss with FocalLoss
#       2. Run the training process 
#       3. 



def cal_performance(pred_linkage, gt_linkage, ignore_index=2, focal_loss=True, 
                    seq_weight=1.0, pos_weight=1.0,  
                    smoothing=False):
    bs, seq_len, dim = pred_linkage.shape
    # link_mask = gt_linkage.eq(1)
    # non_link_mask = gt_linkage.eq(0)
    non_pad_mask = gt_linkage.ne(ignore_index)

    # print("No equal to 2 sum: {}".format(non_pad_mask.sum()))
    # from collections import Counter
    # print("\n {} \t {}".format(gt_linkage.shape, pred_linkage.shape))
    # print("GT counter: {}".format(Counter(gt_linkage.reshape(-1).cpu().numpy())))
    _pred = pred_linkage[non_pad_mask].contiguous().view(-1)
    _gt = gt_linkage[non_pad_mask].contiguous().view(-1)
    # print("Original linkages: {}".format(len(_pred)))
    # print("Collected linkages: {}".format(len(_gt)))

    focal_bce_loss = BCE_FocalLoss_withLogits(gamma=2, alpha=0.5, reduction='mean')
    loss = focal_bce_loss(_pred, _gt)
    
    # pred_linking = pred_linkage[link_mask].view(-1)
    # pred_non_linking = pred_linkage[non_link_mask].view(-1)
    # gt_linking = gt_linkage[link_mask].view(-1)
    # gt_non_linking = gt_linkage[non_link_mask].view(-1)

    pred_linking = (torch.sigmoid(_pred) > 0.5).long()
    # pred_non_linking = (torch.sigmoid(pred_non_linking) < 0.5).int()

    # link_num = (pred_linking == gt_linking).sum()
    # non_link_num = (pred_non_linking == gt_non_linking).sum()
    tn, fp, fn, tp = confusion_matrix(_gt.cpu(), pred_linking.cpu()).ravel()
    total = tn + fp + fn + tp

    # fpr, tpr, thresholds = roc_curve(_gt.cpu(), pred_linking.cpu(), pos_label=1)
    # auc_score = metrics.auc(fpr, tpr)
    # acc = (tp + tn) / total
    # precision = (tp) / (tp + fn)
    # recall = (tp) / (tp + fp)

    return _pred, _gt, loss, tn, fp, fn, tp, total


class BCE_FocalLoss_withLogits(nn.Module):
    def __init__(self, gamma=2, alpha=0.5, reduction='mean') -> None:
        super(BCE_FocalLoss_withLogits, self).__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    # def forward(self, logits, target):
    #     logits = torch.sigmoid(logits)
    #     alpha = self.alpha
    #     gamma = self.gamma
    #     loss = (-1 * alpha * (1 - logits) ** gamma) * target * torch.log(logits) - \
    #             (1 - alpha) * logits ** gamma * (1 - target) * torch.log(1 - logits)
    #     if self.reduciton == 'mean':
    #         loss = loss.mean()
    #     elif self.reduciton == 'sum':
    #         loss = loss.sum()
    #     else:
    #         raise ValueError
    #     return loss 
    
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


def train_epoch(linkage_model, training_data, optimizer, opt, device, smoothing):
    ''' Epoch operation in training phase'''

    linkage_model.train()
    total_loss, link_acc, non_linl_acc = 0, 0, 0
    tp_all, fp_all, fn_all, tn_all = 0, 0, 0, 0

    all_pred_scores = []
    all_gts = []

    desc = '  - (Training)   '
    for batch_data in tqdm(training_data, mininterval=2, desc=desc, leave=False):

        # prepare data
        seq_data_array = batch_data[0].to(torch.float32).cuda(device=device_ids[0])     #.to(device)
        labels = batch_data[1].to(torch.float32).cuda(device=device_ids[0])             # batch x seq_len(512) x 13
                                                                                        # Need to add a BOS token
        amino_nums = batch_data[2]

        src_mask = get_pad_mask(seq_data_array[:, :, 0], pad_idx=0)                     # get mask

        # forward
        optimizer.zero_grad()
        pred_linkage = linkage_model(seq_data_array, src_mask)

        # backward and update parameters
        # loss, link_num, non_link_num, len(gt_linking), len(gt_non_linking)
        _pred, _gt, link_loss, tn, fp, fn, tp, total = \
            cal_performance(pred_linkage, labels, ignore_index=2, focal_loss=True)

        loss = link_loss
        loss.backward()

        # check gradient ---- print
        # for name, params in linkage_model.named_parameters():
        #     print("\n--> name: ", name)
        #     print("    grad_value mean: {}\t std: {}".format(torch.mean(params.grad), torch.std(params.grad)))
        
        optimizer.step_and_update_lr()

        # note keeping
        # all_pred_scores += _pred.cpu().detach().numpy().tolist()
        # all_gts += _gt.cpu().detach().numpy().tolist()
        
        total_loss += loss.item()
        tp_all += tp
        tn_all += tn
        fp_all += fp
        fn_all += fn
    
    # fpr, tpr, thres = metrics.roc_curve(all_gts, all_pred_scores)
    # auc_score = metrics.auc(fpr, tpr)
    # _precision, _recall, thresholds = precision_recall_curve(all_gts, all_pred_scores)
    # _PR_AUC = metrics.auc(_precision, _recall)
    # print("        ---> AUC: {}\t PR-AUC: {}".format(auc_score, _PR_AUC))

    total = tp_all + tn_all + fp_all +fn_all
    acc = (tp_all + tn_all) / total
    precision_all = tp_all / (tp_all + fp_all)
    recall_all = tp_all / (tp_all + fn_all)
    return total_loss, tp_all, fn_all, fp_all, tn_all, total, acc, precision_all, recall_all


def eval_epoch(linkage_model, valid_data,  device=device, phase="Validation"):
    ''' Epoch operation in training phase'''

    linkage_model.eval()
    total_loss, link_acc, non_linl_acc = 0, 0, 0
    tp_all, fp_all, fn_all, tn_all = 0, 0, 0, 0

    desc = '  - (Validing)   '
    with torch.no_grad():
        for batch_data in tqdm(valid_data, mininterval=2, desc=desc, leave=False):

            # prepare data
            seq_data_array = batch_data[0].to(torch.float32).cuda(device=device_ids[0])     #.to(device)
            labels = batch_data[1].to(torch.float32).cuda(device=device_ids[0])             # batch x seq_len(512) x 13
                                                                                        # Need to add a BOS token
            amino_nums = batch_data[2]

            src_mask = get_pad_mask(seq_data_array[:, :, 0], pad_idx=0)                     # get mask

            # forward
            pred_linkage = linkage_model(seq_data_array, src_mask)

            # backward and update parameters
            # loss, link_num, non_link_num, len(gt_linking), len(gt_non_linking)
            _pred, _gt, link_loss, tn, fp, fn, tp, total = \
                cal_performance(pred_linkage, labels, ignore_index=2, focal_loss=True)

            loss = link_loss

            # note keeping
            total_loss += loss.item()
            tp_all += tp
            tn_all += tn
            fp_all += fp
            fn_all += fn

        total = tp_all + tn_all + fp_all +fn_all
        acc = (tp_all + tn_all) / total
        precision_all = tp_all / (tp_all + fp_all)
        recall_all = tp_all / (tp_all + fn_all)
        return total_loss, tp_all, fn_all, fp_all, tn_all, total, acc, precision_all, recall_all


def train(model, training_data, validation_data, test_data, optimizer, cfg, smoothing, device=device):
    # Use tensorboard to plot curves, e.g. perplexity, accuracy, learning rate
    if cfg.use_tb:
        print("[Info] Use Tensorboard")
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(cfg.output_dir, 'tensorboard'))
    
    log_train_file = os.path.join(cfg.output_dir, 'train.log')
    log_valid_file = os.path.join(cfg.output_dir, 'valid.log')

    print('[Info] Training performance will be written to file: {} and {}'.format(
        log_train_file, log_valid_file))
    
    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,accuracy\n')
        log_vf.write('epoch,loss,accuracy\n')
    
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
        train_loss, train_tp, train_fn, train_fp, train_tn, train_total, train_acc, train_precision, train_recall = \
                train_epoch(model, training_data, optimizer, cfg, device, smoothing=cfg.label_smoothing)

        lr = optimizer._optimizer.param_groups[0]['lr']
        print_performances('Training',  train_loss, train_tp, train_fn, train_fp, train_tn, train_total,
                                        train_acc, train_precision, train_recall, 
                                        start, lr)

        start = time.time()
        valid_loss, valid_tp, valid_fn, valid_fp, valid_tn, valid_total, valid_acc, valid_precision, valid_recall = \
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
            log_tf.write('{epoch},{loss: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=train_loss,
                accu=100*train_acc))
            log_vf.write('{epoch},{loss: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=valid_loss,
                accu=100*train_acc))

        if cfg.use_tb:
            tb_writer.add_scalars('ppl', {'train': train_loss, 'val': valid_loss}, epoch_i)
            tb_writer.add_scalars('accuracy', {'train': train_acc*100, 'val': train_acc*100}, epoch_i)
            tb_writer.add_scalar('learning_rate', lr, epoch_i)
        
        # if (epoch_i % 5 == 0):
        #     start = time.time()
        #     test_loss, test_pos_loss, test_seq_loss, test_accu, test_word_total = eval_epoch(model, test_data, device, phase="Testing")
        #     # test_seq_ppl = math.exp(min(test_seq_loss, 100))
        #     # test_pos_ppl = math.exp(min(test_pos_loss, 100))
        #     test_seq_ppl = min(test_seq_loss, 100)
        #     test_pos_ppl = min(test_pos_loss, 100)
        #     print_performances('Testing', test_seq_ppl, test_accu, test_pos_ppl, test_word_total, start, lr)

def test_valid(i):
    valid_set = LinkageSet(index_csv='../datas/tracing_data2/valid.csv', using_gt=False)
    valid_loader = DataLoader(valid_set, shuffle=True, batch_size=cfg.bacth_size * len(device_ids))

    gt_non_pad_nums = 0
    for batch_data in tqdm(valid_loader, mininterval=2, desc="test_valid", leave=False):
         # prepare data
        seq_data_array = batch_data[0].to(torch.float32).cuda(device=device_ids[0])     #.to(device)
        labels = batch_data[1].to(torch.float32).cuda(device=device_ids[0])             # batch x seq_len(512) x 13

        amino_nums = batch_data[2]
        print(amino_nums)

        non_pad_mask = labels.ne(2)
        non_pad_gt = labels[non_pad_mask]
        gt_non_pad_nums += len(non_pad_gt.contiguous().view(-1))
    print("In {} test, the non-pad-part nums are: {}".format(i, gt_non_pad_nums))


def main():
    cfg = Config()
    # transformer = Transformer(22, 22).to(device)
    linkage_model = LinkageFormer(n_layers=2)
    linkage_model = torch.nn.DataParallel(linkage_model, device_ids=device_ids)
    linkage_model = linkage_model.cuda(device=device_ids[0])        

    optimizer = ScheduledOptim(
        optim.Adam(linkage_model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        cfg.lr_mul, cfg.d_model, cfg.n_warmup_steps)
    

    train_set = LinkageSet(index_csv='../datas/tracing_data2/train.csv', using_gt=False, shuffle=False, random_crop=True, crop_bins=8)
    valid_set = LinkageSet(index_csv='../datas/tracing_data2/valid.csv', using_gt=False, shuffle=False, random_crop=True, crop_bins=8)
    test_set  = LinkageSet(index_csv='../datas/tracing_data2/test.csv',  using_gt=False, shuffle=False)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=cfg.bacth_size * len(device_ids), num_workers=4)
    valid_loader = DataLoader(valid_set, shuffle=True, batch_size=cfg.bacth_size * len(device_ids), num_workers=4)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=cfg.bacth_size * len(device_ids), num_workers=4)
    train(linkage_model, train_loader, valid_loader, test_loader, optimizer, cfg, smoothing=False, device=device)
    


if __name__ == "__main__":
    # model = Transformer(n_src_vocab=22,  n_trg_vocab=22)
    main()

    # for i in range(1000):
    #     test_valid(i)--


    # gpu_id = "0, 1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # print('gpu ID is ', str(gpu_id))
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # the_dataset = AminoFeatureDataset(index_csv='../datas/tracing_data/test.csv')
    # the_loader  = DataLoader(the_dataset, batch_size=1)

    # encoder = Encoder(n_amino_feature=22, d_amino_vec=8, n_layers=6, n_head=8, d_k=512, d_v=512,
    #                         d_model=512, d_inner=2048, pad_idx=0, dropout=0.1).to(device)
    # decoder = Decoder(n_amino_feature=22, d_amino_vec=8, n_layers=6, n_head=8, d_k=512, d_v=512,
    #                     d_model=512, d_inner=2048, pad_idx=0, dropout=0.1).to(device)
    # model = Transformer(n_src_vocab=22, n_trg_vocab=22).to(device)

    # linkage_model = LinkageFormer().to(device)
    # the_dataset = LinkageSet(index_csv='../datas/tracing_data/test.csv', using_gt=False)
    # the_loader  = DataLoader(the_dataset, batch_size=1)
    # encoder = Encoder(n_amino_feature=21, d_amino_vec=8, n_layers=6, n_head=8, d_k=512, d_v=512,
    #                         d_model=512, d_inner=2048, pad_idx=0, dropout=0.1).to(device)

    # for idx, data in enumerate(the_loader):
    #     seq_data_array = data[0].to(torch.float32).to(device)
    #     print("Encoder Seq shape: ", seq_data_array.shape)
    #     labels = data[1].to(torch.float32).to(device)
    #     print("Decoder Seq shape: ", labels.shape)
    #     amino_nums = data[2]
    #     print("amino nums: ", amino_nums)
    #     # print(seq_data_array)
    #     print(labels)
        
    #     src_mask = get_pad_mask(seq_data_array[:, :, 0], pad_idx=0)                 # TODO: check masks

    #     output = linkage_model(seq_data_array, src_mask)
    #     print(output)
    #     print(output.shape)

    #     loss = cal_performance(output, labels)
    #     print(loss.item())
    #     break