
"""
    考虑方法合理性
    考虑这样一种可能: 将 tracing type, 和 tracing position 分割开
                    对于给定的集合,  先预测其  type-seq
                    然后通过后处理等方法， 得到整个的sequence prediction
"""
from numpy import diag_indices
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy, math, os
from amino_fea_loader import AminoFeatureDataset
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0


    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class Encoder_Embedding(nn.Module):
    def __init__(self, fea_dim, max_seq_len) -> None:
        super().__init__()
        self.fea_dim = fea_dim
        self.max_seq_len = max_seq_len
    
        self.embeddings = nn.Parameter(torch.Tensor(1, self.max_seq_len, self.fea_dim))
        nn.init.xavier_uniform_(self.embeddings)
    
    def cls(self):
        print(self.embeddings.shape)
        print(self.embeddings)

        batch_size = 10
        kk = self.embeddings.repeat(batch_size, 1, 1, )
        print(kk.shape)
        print(kk)
    
    def forward(self, X):
        factor = 1.0
        embds = self.embeddings.repeat(X.shape[0], 1, 1)
        return X*factor + embds

        

class Decoder_Embedding(nn.Module):
    def __init__(self, fea_dim, max_seq_len, usingPos=False) -> None:
        super().__init__()
        self.fea_dim = fea_dim
        self.max_seq_len = max_seq_len

        if usingPos:
            self.embeddings = self.position_embedding()
        else:
            self.embeddings = nn.Parameter(torch.Tensor(1, self.max_seq_len, self.fea_dim))
            nn.init.xavier_uniform_(self.embeddings)
    
    def position_embedding(self):
        pe = torch.zeros(self.max_seq_len, self.fea_dim)            # max_len x fea_dim
        position = torch.arange(0, self.max_seq_len).unsqueeze(1)   # 1 * max_len
        div_term = torch.exp(torch.arange(0, self.fea_dim, 2) * 
                             -(math.log(10000.0) / self.fea_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                                        # size: (1, max_len, fea_dim)
        return pe


    def forward(self, X):
        factor = 1.0
        embds = self.embeddings.repeat(X.shape[0], 1, 1)
        return X*factor + embds


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, dmodel, d_k, d_v, dropout=0.2) -> None:
        super().__init__()
        self.dmodel = dmodel
        self.d_k = d_k
        self.d_v = d_v
        self.h = heads
        self.q_linear = nn.Linear(dmodel, heads * d_k, bias=False)
        self.k_linear = nn.Linear(dmodel, heads * d_k, bias=False)
        self.v_linear = nn.Linear(dmodel, heads * d_v, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(heads * d_v, dmodel)
        self.layerNorm = nn.LayerNorm(dmodel, eps=1e-6)
    
    
    def forward(self, Q, K, V, attn_mask=None):
        bs = Q.size(0)

        residual = Q

        # perform batch linear operation and split into h heads
        # [batch_size * len_q * n_heads * hidden_dim]
        K_ = self.k_linear(K).view(bs, -1, self.h, self.d_k)
        Q_ = self.q_linear(Q).view(bs, -1, self.h, self.d_k)
        V_ = self.v_linear(V).view(bs, -1, self.h, self.d_v)

        K_ = K_.transpose(1, 2)         # [batch_size * n_heads * len_q * hidden_dim]
        Q_ = Q_.transpose(1, 2)
        V_ = V_.transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.h, 1, 1)          # # For head axis broadcasting.
        attn = torch.matmul(Q_, K_.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask==0, -1e9)                         # DY: May some bugs here for mask broadcasting error?
        attn = F.softmax(attn, dim=-1)
        # attn = self.dropout(attn)

        output = torch.matmul(attn, V_).transpose(
            1, 2).contiguous().view(bs, -1, self.h * self.d_k)          # Batch x len x (heads * d_model)
        output = self.dropout(self.out_proj(output))
        output = self.layerNorm(output + residual)

        return output, attn


class FFN(nn.Module):
    def __init__(self, dmodel, dff, dropout=0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dmodel, dff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dff, dmodel)
        self.layerNorm = nn.LayerNorm(dmodel, eps=1e-6)
    
    def forward(self, x):
        residual = x
        x = self.linear2(F.relu(self.linear1(x)))
        x = self.dropout(x)
        x = self.layerNorm(x + residual)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.ffn = FFN(d_model, d_inner, dropout=dropout)
    
    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, attn_mask=slf_attn_mask
        )
        enc_output = self.ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1) -> None:
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.ffn = FFN(d_model, d_inner, dropout=dropout)
    
    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, attn_mask=slf_attn_mask
        ) 
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, attn_mask=dec_enc_attn_mask
        )
        dec_output = self.ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    bs, len_q, len_dims = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_q, len_q, ), device=seq.device), diagonal=1)
        ).bool()
    return subsequent_mask              # triangle bottom matrix?                             


class Encoder(nn.Module):
    def __init__(self, n_amino_feature, d_amino_vec, n_layers, n_head, d_k, d_v,
                 d_model, d_inner, pad_idx, dropout=0.1, n_posititon=None, scale_emb=False, max_len=512) -> None:
        super().__init__()

        self.src_amino_vec = nn.Embedding(n_amino_feature, d_amino_vec, padding_idx=pad_idx)    # n_amino_feature set to 22, d_amino_vec set to 8
        self.position_eonc = Encoder_Embedding(fea_dim=d_model, max_seq_len=max_len)            # TODO: change the Encoding format
        self.linearB = nn.Linear(20, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model
    
    def forward(self, src_seq, src_mask, return_attns = False):
        bs, seq_len, dims = src_seq.shape
        assert dims == 13
        # embeding type and concat
        amino_type_vec = self.src_amino_vec(src_seq[:, :, 0].long())
        fea_vec = torch.cat([amino_type_vec, src_seq[:, :, 1:]], dim=2)

        enc_slf_attn_list = []

        # enc_output = self.src_amino_vec(src_seq)
        enc_output = self.linearB(fea_vec)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(enc_output)
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
        
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


class Decoder(nn.Module):
    def __init__(self, n_amino_feature, d_amino_vec, n_layers, n_head, d_k, d_v,
                       d_model, d_inner, pad_idx, dropout=0.1, n_posititon=None, scale_emb=False, max_len=512) -> None:
        super().__init__()
        self.src_amino_vec = nn.Embedding(n_amino_feature, d_amino_vec, padding_idx=pad_idx)    # n_amino_feature set to 22, d_amino_vec set to 8
        self.position_enc = PositionalEncoding(d_model, n_position=512)
        self.linearB = nn.Linear(20, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout) for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model
    
    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):
        bs, seq_len, dims = trg_seq.shape
        assert dims == 13
        amino_type_vec = self.src_amino_vec(trg_seq[:, :, 0].long())
        fea_vec = torch.cat([amino_type_vec, trg_seq[:, :, 1:]], dim=2)
        
        dec_slf_attn_list, dec_enc_attn_list = [], []

        dec_output = self.linearB(fea_vec)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)
        
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []
        
        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output



class Transformer(nn.Module):
    def __init__(self, 
                n_src_vocab, n_trg_vocab, src_pad_idx=0, trg_pad_idx=0,                 
                d_amino_type_vec=8, d_model=512, d_inner=2048,
                n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=512,
                trg_emb_prj_weight_sharing=False, 
                emb_src_trg_weight_sharing=False,
                scale_emb_or_prj='prj') -> None:
        super().__init__()
        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx               # usually, the src_pad_idx is none

        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here:
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication
        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False            # scale factor
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False       # 
        self.d_model = d_model

        # for encoder, in fact, there is no use of position-encoding, I just use a simple linear parameters 
        # initialized with xavier form.
        self.encoder = Encoder(
            n_amino_feature=n_src_vocab, d_amino_vec=d_amino_type_vec, n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            d_model=d_model, d_inner=d_inner, pad_idx=src_pad_idx, dropout=dropout, n_posititon=n_position,         
            scale_emb=scale_emb,
        )

        self.decoder = Decoder(
            n_amino_feature=n_trg_vocab, d_amino_vec=d_amino_type_vec, n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            d_model=d_model, d_inner=d_inner, pad_idx=trg_pad_idx, dropout=dropout, n_posititon=n_position, 
            scale_emb=scale_emb
        )

        self.trg_amino_type_prj = nn.Linear(d_model, 23, bias=False)        # project to 22D or 23D ?
        self.trg_amino_pos_prj = nn.Linear(d_model, 12, bias=False)
        # self.linear_clf = nn.Linear(d_amino_type_vec, 22)
        # self.linear_pos = nn.linear(12, 12)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # assert d_model == d_word_vec

        # if trg_emb_prj_weight_sharing:
        #     # Share the weight between target word embedding & last dense layer
        #     self.trg_word_prj.weight = self.decoder.src_amino_vec.weight
        
        if emb_src_trg_weight_sharing:
            # Share the weight between target word embedding & src word embedding
            self.encoder.src_amino_vec.weight = self.decoder.src_amino_vec.weight
    
    def forward(self, src_seq, trg_seq):
        src_mask = get_pad_mask(src_seq[:, :, 0], self.src_pad_idx)              # for our task, there is no need for src_mask
        trg_mask = get_pad_mask(trg_seq[:, :, 0], self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output = self.encoder(src_seq, src_mask)
        dec_output = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        seq_logit = self.trg_amino_type_prj(dec_output)
        seq_pos = self.trg_amino_pos_prj(dec_output)
        if self.scale_prj:
            seq_logit *= self.d_model ** 0.5
        
        amino_seq = seq_logit.view(-1, seq_logit.size(1), seq_logit.size(2))
        amino_pos = seq_pos
        return amino_seq, amino_pos
        # return seq_logit.view(-1, seq_logit.size(2)), seq_pos.view(-1, seq_pos.size[2])
        

if __name__ == '__main__': 
    # embs = Encoder_Embedding(fea_dim=32, max_seq_len=512)
    # embs.cls()

    gpu_id = "0, 1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print('gpu ID is ', str(gpu_id))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    the_dataset = AminoFeatureDataset(index_csv='../datas/tracing_data/test.csv')
    the_loader  = DataLoader(the_dataset, batch_size=1)

    encoder = Encoder(n_amino_feature=22, d_amino_vec=8, n_layers=6, n_head=8, d_k=512, d_v=512,
                            d_model=512, d_inner=2048, pad_idx=0, dropout=0.1).to(device)
    decoder = Decoder(n_amino_feature=22, d_amino_vec=8, n_layers=6, n_head=8, d_k=512, d_v=512,
                        d_model=512, d_inner=2048, pad_idx=0, dropout=0.1).to(device)
    model = Transformer(n_src_vocab=22, n_trg_vocab=22).to(device)

    for idx, data in enumerate(the_loader):
        seq_data_array = data[0].to(torch.float32).to(device)
        # print("Encoder Seq shape: ", seq_data_array.shape)
        labels = data[1].to(torch.float32).to(device)
        # print("Decoder Seq shape: ", labels.shape)
        # print(seq_data_array)
        # print(labels)
        
        src_mask = get_pad_mask(seq_data_array[:, :, 0], pad_idx=0)                 # TODO: check masks
        # print("src_mask shape: ", src_mask.shape)
        # print(src_mask)
        # src_mask = None
        enc_output = encoder(seq_data_array, src_mask)
        # print(enc_output)

        # print("\n******* Decoder Part *******\n")
        trg_mask = get_pad_mask(labels[:, :, 0], pad_idx=0)  & get_subsequent_mask(labels)
        dec_output = decoder(labels, trg_mask, enc_output, src_mask)
        # print("dec_output shape:", dec_output.shape)
        # print(dec_output[0])

        amino_seq, amino_pos =  model(seq_data_array, labels)
        print("\n============Transformer model output:==============")
        print(amino_seq.shape)
        print(amino_seq)
        print(amino_pos.shape)
        print(amino_pos)
        break
