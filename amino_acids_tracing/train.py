import argparse
import os, time, math
from matplotlib.pyplot import clf
from TraceFormer import Transformer, get_pad_mask, get_subsequent_mask, ScheduledOptim
from amino_fea_loader import AminoFeatureDataset
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Config import Config

gpu_id = "0, 1"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
print('gpu ID is ', str(gpu_id))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seq_clf_loss_weight = 1.0
seq_pos_loss_weight = 1.0

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
    amino_pos_loss = cal_pos_loss(_pred_pos, _gt_pos, non_pad_mask=non_pad_mask.view(-1))               # TODO: if need this index？
    # loss = amino_seq_loss * seq_weight +  amino_pos_loss * pos_weight
    # loss /= (seq_weight + pos_weight)
    # print("Losses are:{} \t {}".format(amino_pos_loss.item(), amino_seq_loss.item()))

    # print("Pred's shape:", _pred_seq.shape)
    pred = _pred_seq.max(1)[1]
    # print("Pred's shape:", pred.shape)
    gt_type = _gt_seq.contiguous().view(-1)
    non_pad_mask = gt_seq.ne(0).view(-1)
    n_correct = pred.eq(gt_type).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return amino_seq_loss, amino_pos_loss, n_correct, n_word



def cal_seq_loss(pred_amino_seq, gt_seq, trg_pad_idx):
    # bs, dim = gt_seq.shape
    # print("Amino type pred shape: {}\t{}".format(pred_amino_seq.shape, gt_seq.shape))

    # _pred = pred_amino_seq.view(-1, pred_amino_seq.size(-1))
    # _gt = gt_seq.view(-1).long()
    # print(_pred.shape, _gt.shape)

    clf_loss = F.cross_entropy(pred_amino_seq, gt_seq, ignore_index=0, reduction="sum").to(device)
    return clf_loss


def cal_pos_loss(pred_amino_pos, gt_pos, non_pad_mask=None, device=device):
    # print("\n _pos_shape: {}\t{}".format(pred_amino_pos.shape, gt_pos.shape))
    # # if trg_pad_idx is not None:
    #     trg_pad_idx = torch.tensor(trg_pad_idx).detach().cpu().numpy()
    #     first_pad_idx = np.where(trg_pad_idx == 0)[0][0]
    # else:
    #     first_pad_idx = len(pred_amino_pos)
    # print("\nfirst_pad index:", first_pad_idx, "\n")
    # TODO: No mask here
    # _pred_pos = pred_amino_pos.view(-1, pred_amino_pos.size(-1))
    # _gt_pos = gt_pos.view(-1, gt_pos.size(-1))
    smooth_l1_fn = nn.SmoothL1Loss(reduction='mean', beta=1.0)
    pos_loss = smooth_l1_fn(pred_amino_pos[non_pad_mask], gt_pos[non_pad_mask])
    return pos_loss



def train_epoch(model, training_data, optimizer, opt, device, smoothing):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss, pos_loss, type_loss= 0, 0, 0 
    n_word_total, n_word_correct = 0, 0

    desc = '  - (Training)   '
    for batch_data in tqdm(training_data, mininterval=2, desc=desc, leave=False):

        # prepare data
        seq_data_array = batch_data[0].to(torch.float32).to(device)
        labels = batch_data[1].to(torch.float32).to(device)         # batch x seq_len(512) x 13
        src_mask = get_pad_mask(seq_data_array[:, :, 0], pad_idx=0)
        trg_pad_idx = get_pad_mask(labels[:, :, 0], pad_idx=0)
        trg_mask = trg_pad_idx & get_subsequent_mask(labels)

        # forward
        optimizer.zero_grad()
        pred_seq, pred_pos = model(seq_data_array, labels)

        # backward and update parameters
        seq_typ_loss, seq_pos_loss, n_correct, n_word = cal_performance(
            pred_seq, pred_pos, labels[:, :, 0], labels[:, :, 1:], trg_pad_idx=trg_pad_idx) 

        loss = seq_clf_loss_weight * seq_typ_loss + seq_pos_loss_weight * seq_pos_loss
        loss /= (seq_clf_loss_weight + seq_pos_loss_weight)
        loss.backward()
        optimizer.step_and_update_lr()

        # note keeping
        n_word_total += n_word
        n_word_correct += n_correct
        type_loss += seq_typ_loss.item()
        pos_loss += seq_pos_loss.item()
        total_loss += loss.item()

    loss_per_word = type_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    pos_loss_per_amino = pos_loss / n_word_total
    return loss, pos_loss_per_amino, loss_per_word, accuracy


def eval_epoch(model, validation_data, phase="Validation", device=device):
    model.eval()
    total_loss, pos_loss, type_loss= 0, 0, 0 
    n_word_total, n_word_correct = 0, 0

    desc = '  - ({})   '.format(phase)
    with torch.no_grad():
        for batch_data in tqdm(validation_data, mininterval=2, desc=desc, leave=False):
            # prepare data
            seq_data_array = batch_data[0].to(torch.float32).to(device)
            labels = batch_data[1].to(torch.float32).to(device)         # batch x seq_len(512) x 13
            src_mask = get_pad_mask(seq_data_array[:, :, 0], pad_idx=0)
            trg_pad_idx = get_pad_mask(labels[:, :, 0], pad_idx=0)
            trg_mask = trg_pad_idx & get_subsequent_mask(labels)

            # forward
            pred_seq, pred_pos = model(seq_data_array, labels)

            # backward and update parameters
            seq_typ_loss, seq_pos_loss, n_correct, n_word = cal_performance(
                pred_seq, pred_pos, labels[:, :, 0], labels[:, :, 1:], trg_pad_idx) 
            loss = seq_clf_loss_weight * seq_typ_loss + seq_pos_loss_weight * seq_pos_loss
            loss /= (seq_clf_loss_weight + seq_pos_loss_weight)
            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            type_loss += seq_typ_loss.item()
            pos_loss += seq_pos_loss.item()
            total_loss += loss.item()

    loss_per_word = type_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    pos_loss_per_amino = pos_loss / n_word_total
    return loss, pos_loss_per_amino, loss_per_word, accuracy


def train(model, training_data, validation_data,  optimizer, cfg, smoothing, device=device):
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
        log_tf.write('epoch,loss,ppl,accuracy\n')
        log_vf.write('epoch,loss,ppl,accuracy\n')
    
    def print_performances(header, ppl, accu, pos_ppl, start_time, lr):
        print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, pos_ppl:{pos_ppl:8.5f}\
                lr: {lr:8.5f}, '\
                'elapse: {elapse:3.3f} min'.format(
                  header=f"({header})", ppl=ppl,
                  accu=100*accu, pos_ppl=pos_ppl, elapse=(time.time()-start_time)/60, lr=lr))
    
    valid_losses = []
    for epoch_i in range(cfg.num_epochs):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_pos_loss, train_seq_loss, train_accu = train_epoch(
                                                            model, training_data, optimizer, cfg, device, smoothing=cfg.label_smoothing)
        # train_ppl = math.exp(min(train_loss, 100))
        train_seq_ppl = math.exp(min(train_seq_loss, 100))
        train_pos_ppl = math.exp(min(train_pos_loss, 100))
        # Current learning rate
        lr = optimizer._optimizer.param_groups[0]['lr']
        print_performances('Training', train_seq_ppl, train_accu, train_pos_ppl, start, lr)

        start = time.time()
        valid_loss, val_pos_loss, val_seq_loss, valid_accu = eval_epoch(model, validation_data, device, device)
        # valid_ppl = math.exp(min(valid_loss, 100))
        valid_seq_ppl = math.exp(min(val_seq_loss, 100))
        valid_pos_ppl = math.exp(min(valid_accu, 100))
        print_performances('Validation', valid_seq_ppl, valid_accu, valid_pos_ppl, start, lr)

        valid_losses += [valid_loss]

        checkpoint = {'epoch': epoch_i, 'settings': cfg, 'model': model.state_dict()}

        if cfg.save_mode == 'all':
            model_name = 'model_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
            torch.save(checkpoint, model_name)
        elif cfg.save_mode == 'best':
            model_name = 'model.chkpt'
            if valid_loss <= min(valid_losses):
                torch.save(checkpoint, os.path.join(cfg.output_dir, model_name))
                print('    - [Info] The checkpoint file has been updated.')
        
        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=train_loss,
                ppl=train_seq_ppl, accu=100*train_accu))
            log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=valid_loss,
                ppl=valid_seq_ppl, accu=100*valid_accu))

        if cfg.use_tb:
            tb_writer.add_scalars('ppl', {'train': train_seq_ppl, 'val': valid_seq_ppl}, epoch_i)
            tb_writer.add_scalars('accuracy', {'train': train_accu*100, 'val': valid_accu*100}, epoch_i)
            tb_writer.add_scalar('learning_rate', lr, epoch_i)


def main():
    cfg = Config()
    transformer = Transformer(22, 22).to(device)
    optimizer = ScheduledOptim(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
        cfg.lr_mul, cfg.d_model, cfg.n_warmup_steps)
    

    train_set = AminoFeatureDataset(index_csv='../datas/tracing_data/train.csv')
    valid_set = AminoFeatureDataset(index_csv='../datas/tracing_data/valid.csv')
    test_set = AminoFeatureDataset(index_csv='../datas/tracing_data/test.csv')

    train_loader = DataLoader(train_set, shuffle=True, batch_size=cfg.bacth_size)
    valid_loader = DataLoader(valid_set, shuffle=True, batch_size=cfg.bacth_size)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=cfg.bacth_size)
    train(transformer, train_loader, valid_loader, optimizer, cfg, smoothing=False, device=device)
    


if __name__ == "__main__":
    # model = Transformer(n_src_vocab=22,  n_trg_vocab=22)
    main()


    
