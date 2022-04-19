import argparse
import os, time, math
from turtle import forward
from TraceFormer import Transformer, get_pad_mask, get_subsequent_mask, ScheduledOptim, Encoder, Decoder, LinkageFormer
from amino_fea_loader import AminoFeatureDataset, LinkageSet
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Config import Config


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
    non_pad_mask = gt_linkage.ne(ignore_index)
    print("No equal to 2 sum: {}".format(non_pad_mask.sum()))
    from collections import Counter
    print("\n {} \t {}".format(gt_linkage.shape, pred_linkage.shape))
    print("GT counter: {}".format(Counter(gt_linkage.reshape(-1).cpu().numpy())))
    _pred = pred_linkage[non_pad_mask].view(-1)
    _gt = gt_linkage[non_pad_mask].view(-1)
    print("Original linkages: {}".format(len(_pred)))
    print("Collected linkages: {}".format(len(_gt)))

    focal_bce_loss = BCE_FocalLoss_withLogits(reduction='sum')
    loss = focal_bce_loss(_pred, _gt)

    return loss


class BCE_FocalLoss_withLogits(nn.Module):
    def __init__(self, gamma=2, alpha=0.5, reduction='mean') -> None:
        super(BCE_FocalLoss_withLogits, self).__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.reduciton = reduction

    def forward(self, logits, target):
        logits = torch.sigmoid(logits)
        alpha = self.alpha
        gamma = self.gamma
        loss = (-1 * alpha * (1 - logits) ** gamma) * target * torch.log(logits) - \
                (1 - alpha) * logits ** gamma * (1 - target) * torch.log(1 - logits)
        if self.reduciton == 'mean':
            loss = loss.mean()
        elif self.reduciton == 'sum':
            loss = loss.sum()
        else:
            raise ValueError
        return loss 


def train_epoch(model, training_data, optimizer, opt, device, smoothing):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss, pos_loss, type_loss= 0, 0, 0 
    n_word_total, n_word_correct = 0, 0

    desc = '  - (Training)   '
    for batch_data in tqdm(training_data, mininterval=2, desc=desc, leave=False):

        # prepare data
        seq_data_array = batch_data[0].to(torch.float32).cuda(device=device_ids[0])      #.to(device)
        labels = batch_data[1].to(torch.float32).cuda(device=device_ids[0])             # batch x seq_len(512) x 13
                                                                                        # Need to add a BOS token
        # add BOS token
        # temp = labels[:, :511, :]
        # bos = torch.tensor([0] * 13, device=device); bos[0] = 22
        # labels[:, 0, :] = bos
        # labels[:, 1:, :] = temp

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
        loss /= (seq_clf_loss_weight + seq_pos_loss_weight)                                     # if needed ?
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
    return total_loss, pos_loss_per_amino, loss_per_word, accuracy, n_word_total


def eval_epoch(model, validation_data, device=device, phase="Validation"):
    model.eval()
    total_loss, pos_loss, type_loss= 0, 0, 0 
    n_word_total, n_word_correct = 0, 0

    desc = '  - ({})   '.format(phase)
    with torch.no_grad():
        for batch_data in tqdm(validation_data, mininterval=2, desc=desc, leave=False):
            # prepare data
            seq_data_array = batch_data[0].to(torch.float32).cuda(device=device_ids[0])   
            labels = batch_data[1].to(torch.float32).cuda(device=device_ids[0])            # batch x seq_len(512) x 13
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
    return total_loss, pos_loss_per_amino, loss_per_word, accuracy, n_word_total


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
        log_tf.write('epoch,loss,seq_loss,pos_loss,accuracy\n')
        log_vf.write('epoch,loss,seq_loss,pos_loss,accuracy\n')
    
    def print_performances(header, seq_loss, accu, pos_loss, word_total, start_time, lr):
        print('  - {header:12} seq_loss: {seq_loss: 8.5f}, pos_loss:{pos_loss:8.5f}, accuracy: {accu:3.3f} %, \
                word_total:{word_total:6},\
                lr: {lr:8.5f}, '\
                'elapse: {elapse:3.3f} min'.format(
                  header=f"({header})", seq_loss=seq_loss, word_total=word_total,
                  accu=100*accu, pos_loss=pos_loss, elapse=(time.time()-start_time)/60, lr=lr))
    
    valid_losses = []
    for epoch_i in range(cfg.num_epochs):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_pos_loss, train_seq_loss, train_accu, train_word_total = \
                        train_epoch(model, training_data, optimizer, cfg, device, smoothing=cfg.label_smoothing)
        # train_ppl = math.exp(min(train_loss, 100))
        # train_seq_ppl = math.exp(min(train_seq_loss, 100))
        # train_pos_ppl = math.exp(min(train_pos_loss, 100))        
        train_seq_ppl = min(train_seq_loss, 100)
        train_pos_ppl = min(train_pos_loss, 100)
        # Current learning rate
        lr = optimizer._optimizer.param_groups[0]['lr']
        print_performances('Training', train_seq_ppl, train_accu, train_pos_ppl, train_word_total, start, lr)

        start = time.time()
        valid_loss, val_pos_loss, val_seq_loss, valid_accu, valid_word_total = eval_epoch(model, validation_data, device, phase="Validation")
        # valid_ppl = math.exp(min(valid_loss, 100))
        # valid_seq_ppl = math.exp(min(val_seq_loss, 100))
        # valid_pos_ppl = math.exp(min(valid_accu, 100))
        valid_seq_ppl = min(val_seq_loss, 100)
        valid_pos_ppl = min(val_pos_loss, 100)
        print_performances('Validation', valid_seq_ppl, valid_accu, valid_pos_ppl, valid_word_total, start, lr)

        valid_losses += [valid_loss]

        checkpoint = {'epoch': epoch_i, 'settings': cfg, 'model': model.state_dict()}

        if cfg.save_mode == 'all':
            model_name = 'model_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
            torch.save(checkpoint, model_name)
            if valid_accu == 1.0:
                best_model_name = "model_accu_100.000_pos_los{}.chkpt".format(val_pos_loss)
                torch.save(checkpoint, best_model_name)
        elif cfg.save_mode == 'best':
            model_name = 'model.chkpt'
            if valid_loss <= min(valid_losses):
                torch.save(checkpoint, os.path.join(cfg.output_dir, model_name))
                print('    - [Info] The checkpoint file has been updated.')
        
        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss: 8.5f},{seq_loss: 8.5f},{pos_loss: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=train_loss,
                seq_loss=train_seq_ppl, pos_loss=train_pos_ppl, accu=100*train_accu))
            log_vf.write('{epoch},{loss: 8.5f},{seq_loss: 8.5f},{pos_loss: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=valid_loss,
                seq_loss=valid_seq_ppl, pos_loss=valid_pos_ppl, accu=100*valid_accu))

        if cfg.use_tb:
            tb_writer.add_scalars('ppl', {'train': train_seq_ppl, 'val': valid_seq_ppl}, epoch_i)
            tb_writer.add_scalars('accuracy', {'train': train_accu*100, 'val': valid_accu*100}, epoch_i)
            tb_writer.add_scalar('learning_rate', lr, epoch_i)
        
        if (epoch_i % 5 == 0):
            start = time.time()
            test_loss, test_pos_loss, test_seq_loss, test_accu, test_word_total = eval_epoch(model, test_data, device, phase="Testing")
            # test_seq_ppl = math.exp(min(test_seq_loss, 100))
            # test_pos_ppl = math.exp(min(test_pos_loss, 100))
            test_seq_ppl = min(test_seq_loss, 100)
            test_pos_ppl = min(test_pos_loss, 100)
            print_performances('Testing', test_seq_ppl, test_accu, test_pos_ppl, test_word_total, start, lr)



def main():
    cfg = Config()
    # transformer = Transformer(22, 22).to(device)
    transformer = Transformer(21, 21)
    transformer = torch.nn.DataParallel(transformer, device_ids=device_ids)
    transformer = transformer.cuda(device=device_ids[0])        

    optimizer = ScheduledOptim(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
        cfg.lr_mul, cfg.d_model, cfg.n_warmup_steps)
    

    train_set = AminoFeatureDataset(index_csv='../datas/tracing_data2/train.csv', z_score_coords=False)
    valid_set = AminoFeatureDataset(index_csv='../datas/tracing_data2/valid.csv', z_score_coords=False)
    test_set  = AminoFeatureDataset(index_csv='../datas/tracing_data2/test.csv',  z_score_coords=False)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=cfg.bacth_size * len(device_ids))
    valid_loader = DataLoader(valid_set, shuffle=True, batch_size=cfg.bacth_size * len(device_ids))
    test_loader = DataLoader(test_set, shuffle=False, batch_size=cfg.bacth_size * len(device_ids))
    train(transformer, train_loader, valid_loader, test_loader, optimizer, cfg, smoothing=False, device=device)
    


if __name__ == "__main__":
    # model = Transformer(n_src_vocab=22,  n_trg_vocab=22)
    # main()

    gpu_id = "0, 1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print('gpu ID is ', str(gpu_id))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # the_dataset = AminoFeatureDataset(index_csv='../datas/tracing_data/test.csv')
    # the_loader  = DataLoader(the_dataset, batch_size=1)

    encoder = Encoder(n_amino_feature=22, d_amino_vec=8, n_layers=6, n_head=8, d_k=512, d_v=512,
                            d_model=512, d_inner=2048, pad_idx=0, dropout=0.1).to(device)
    decoder = Decoder(n_amino_feature=22, d_amino_vec=8, n_layers=6, n_head=8, d_k=512, d_v=512,
                        d_model=512, d_inner=2048, pad_idx=0, dropout=0.1).to(device)
    model = Transformer(n_src_vocab=22, n_trg_vocab=22).to(device)

    linkage_model = LinkageFormer().to(device)
    the_dataset = LinkageSet(index_csv='../datas/tracing_data/test.csv', using_gt=False)
    the_loader  = DataLoader(the_dataset, batch_size=1)
    encoder = Encoder(n_amino_feature=21, d_amino_vec=8, n_layers=6, n_head=8, d_k=512, d_v=512,
                            d_model=512, d_inner=2048, pad_idx=0, dropout=0.1).to(device)

    for idx, data in enumerate(the_loader):
        seq_data_array = data[0].to(torch.float32).to(device)
        print("Encoder Seq shape: ", seq_data_array.shape)
        labels = data[1].to(torch.float32).to(device)
        print("Decoder Seq shape: ", labels.shape)
        amino_nums = data[2]
        print("amino nums: ", amino_nums)
        # print(seq_data_array)
        print(labels)
        
        src_mask = get_pad_mask(seq_data_array[:, :, 0], pad_idx=0)                 # TODO: check masks

        output = linkage_model(seq_data_array, src_mask)
        print(output)
        print(output.shape)

        loss = cal_performance(output, labels)
        print(loss.item())
        break