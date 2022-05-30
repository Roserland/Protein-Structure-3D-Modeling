"""
    TODO:   1. Using beam-search to generate
            2. Generate type-seq and correspondingly positions seperately

"""
import os, tqdm, random, json
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader
from TraceFormer import Transformer, LinkageFormer, get_pad_mask, get_subsequent_mask
from amino_fea_loader import AminoFeatureDataset, LinkageSet
from Config import Config
import numpy as np
from linkage_train import cal_performance as _cal_performance
from my_datasets import MLP_Protein
from simple_models import MLP
from PIL import Image

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


def linkage_inference(cfg, device):
    # load model
    linkage_model = LinkageFormer(n_layers=2)
    linkage_model = torch.nn.DataParallel(linkage_model, device_ids=device_ids)

    ckpnt = torch.load(cfg.linkage_model_path)
    _state_dict = ckpnt['model']
    linkage_model.load_state_dict(_state_dict)  

    test_set  = LinkageSet(index_csv='../datas/tracing_data2/test.csv',  using_gt=False)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=1)

    linkage_model.eval()

    total_loss, link_acc, non_linl_acc = 0, 0, 0
    tp_all, fp_all, fn_all, tn_all = 0, 0, 0, 0
    # calculate perfomance
    _preds = np.array([])
    _gts = np.array([])
    for batch_data in tqdm(test_loader, mininterval=2, desc="infer", leave=False):
        # prepare data
        seq_data_array = batch_data[0].to(torch.float32).cuda(device=device_ids[0])     #.to(device)
        labels = batch_data[1].to(torch.float32).cuda(device=device_ids[0])             # batch x seq_len(512) x 13
                                                                                        # Need to add a BOS token
        amino_nums = batch_data[2]
        

        src_mask = get_pad_mask(seq_data_array[:, :, 0], pad_idx=0)                     # get mask

        # forward
        pred_linkage = linkage_model(seq_data_array, src_mask)
        the_pred, the_gt, loss, tn, fp, fn, tp, total = \
            _cal_performance(pred_linkage, labels, ignore_index=2, focal_loss=True)
        _preds = np.concatenate(_preds, the_pred.cpu().detach().numpy)
        _gts = np.concatenate(_gts, the_gt.cpu().detach().numpy)
    
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


class ProteinLinker():
    def __init__(self, model, 
                       fea_src_dir='/mnt/data/zxy/relat_coors_stage3-amino-keypoint-vectors/', 
                       label_src_dir='/mnt/data/zxy/stage3_data/stage3_labels/',
                       max_len=512,
                       pad_index=2, ) -> None:
        super().__init__()

        self.model = model
        self.fea_src_dir = fea_src_dir
        self.label_src_dir = label_src_dir
        self.max_len = max_len
        self.device_ids = [0, 1]

        self.pad_index=pad_index

        self.amino_data_array = None
    

    def load_predicted_aminos(self, protein_id, pad=0, max_len=512, shuffle=False):
        amino_acids_dir = os.path.join(self.fea_src_dir, protein_id)
        amino_file_list = os.listdir(amino_acids_dir)

        random.shuffle(amino_file_list)        

        if len(amino_file_list) > max_len:
            print("The detected amino acids num is larger than {}. Split the set or change the 'max_len' ".format(max_len))

        data_array = np.zeros((max_len, 13)) + pad
        amino_index_list = []
        for i, amino_file in enumerate(amino_file_list):
            data_array[i] = np.load(os.path.join(amino_acids_dir, amino_file)).reshape(-1)
            amino_index_list.append(int(amino_file[:-4].split('_')[1]))
        
        self.amino_data_array = data_array
        self.amino_index_list = amino_index_list
    

    def predict_linkage(self, max_len=512):
        """
            predicted_amino_acids: 2D array -> [amino-1, amino-2, amino-3....]
                                                amino-1: a 13D vector
            
        """
        # detected_nums = len(predicted_amino_acids)

        # if detected_nums > self.max_len:
        #     """
        #     detected amino-acids num is larger than the default nums set by model
        #     So, it's necessary to split the amino-acids into several buckets with overlap.
        #     Then, construt the whole sets with splitted results
        #     """
        #     pass
        # else:
        #     # buckets = [predicted_amino_acids]
        #     pass
            
        input_data = np.zeros((max_len, 13))
        length, fea_dim = self.amino_data_array.shape
        input_data[:length] = self.amino_data_array

        input_data = torch.from_numpy(input_data).to(torch.float32).unsqueeze(0) # .cuda(device=device_ids[0])
        # input_data.cuda(device=self.device_ids[0])

        src_mask = get_pad_mask(input_data[:, :, 0], pad_idx=0)                     # get mask

        pred_linkage = self.model(input_data, src_mask)
        sigmoid_pred = torch.sigmoid(pred_linkage)

        return sigmoid_pred


    def reload_model(model_path):
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


class MLP_Linkage_Predictor():
    def __init__(self, model, model_ckpnt_path=None) -> None:
        self.model = model
        self.cnkpt_path = model_ckpnt_path
        self.fea_src_dir = '/mnt/data/zxy/relat_coors_stage3-amino-keypoint-vectors/'
        self.label_src_dir = '/mnt/data/zxy/stage3_data/stage3_labels/'
        self.label_linkage_dir = '/mnt/data/zxy/stage3_data/stage3_labels/'
        self.image_dst_dir = './output_dir/mlp_res_imgs/'


    def coordinates_normalization(self, datas, mean_std_params_file='../datas/params/coord_mean_std.json'):
        length, dims = datas.shape
        assert dims == 24 
        with open(mean_std_params_file, 'r')  as j:
            _params = json.load(j)
        
        x_mean_std = _params['x_mean_std']                  # TODO: Of cource, there face the same problem, the "X" axis and "Z" axis may be disordered
        y_mean_std = _params['y_mean_std']                  # But the mean and std are not far from in X and Z, so, the impact is limmited in some way
        z_mean_std = _params['z_mean_std']                  # DY: ATTENTION
        
        # with out type info
        datas[:, 0::3] = (datas[:, 0::3] - x_mean_std[0])  / x_mean_std[1]
        datas[:, 1::3] = (datas[:, 1::3] - y_mean_std[0])  / y_mean_std[1]
        datas[:, 2::3] = (datas[:, 2::3] - z_mean_std[0])  / z_mean_std[1]

        return datas
    
    def mask_data(self, data_array):
        w = data_array.shape[0]
        mask = np.ones((w, w)) - np.eye(w)
        return data_array * mask
    
    def predict_amino_linkage(self, protein_id):
        mlp_protein = MLP_Protein(protein_id=protein_id, method='concat', using_whole=True)

        # the data in diag may not the valid data and may cause mistakes
        data_array = mlp_protein.processed_data
        gt_labels = mlp_protein.processed_label
        print("Data_array shape: ", data_array.shape)
        print("gt_labels  shape: ", gt_labels.shape)

        data_array = self.coordinates_normalization(data_array)

        predction = self.model(torch.from_numpy(data_array).float())
        predction = torch.sigmoid(predction).detach().cpu().numpy()

        w = int(len(gt_labels) ** 0.5)
        self.data_width = w

        _pred_image = predction.reshape((w, w))
        print("_pred_image shape: ", _pred_image.shape)
        print("_pred_image:", _pred_image)
        _pred_image = self.mask_data(_pred_image)
        print("_pred_image:", _pred_image)


        # image_path = os.path.join(self.image_dst_dir, protein_id + ".png")
        # img = Image.fromarray(_pred_image)
        # img.save(image_path)

        heatmap_image = (_pred_image * 255).astype(int)
        heatmap_image = Image.fromarray(heatmap_image.astype('uint8'))
        heatmap_image.save(os.path.join(self.image_dst_dir, "{}_heatmap.png".format(protein_id)))
        thres_image  = (_pred_image >= 0.5).astype(int) * 255
        thres_image = Image.fromarray(thres_image.astype('uint8'))
        thres_image.save(os.path.join(self.image_dst_dir, "{}_binary_thres.png".format(protein_id)))

    



if __name__ == '__main__':
    # # function_test()
    print("GPU available: {}".format(torch.cuda.device_count()))
    gpu_id = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print("GPU available: {}".format(torch.cuda.device_count()))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_ids = [0, 1]
    # _inference_test(cfg, device)


    # linkage_model = LinkageFormer(n_layers=2)
    # linkage_model = nn.DataParallel(linkage_model)

    # ckpnt = torch.load(cfg.linkage_model_path)
    # _state_dict = ckpnt['model']
    # linkage_model.load_state_dict(_state_dict)

    # linkage_model.cuda(0)
    
    # # inference
    # print("\nPrediction Linkage...")
    # linkage_predictor = ProteinLinker(linkage_model)
    
    # linkage_predictor.load_predicted_aminos(protein_id="6VV0")
    # link_res = linkage_predictor.predict_linkage()

    # print(linkage_predictor.amino_index_list)
    # the_len = len(linkage_predictor.amino_index_list)
    # print("Detected nums: {}".format(the_len))
    # print(link_res.shape)
    # print(link_res)
    
    # print("Checking")
    # for i in range(1, the_len):
    #     print(link_res[0][i][i-1])
    # print("\nThe padding part")
    # for i in range(the_len, 512):
    #     print(link_res[0][i][i-1])

    # ===============================================================
    # Section 3.
    # MLP Linkage Inference Part
    # Predict affinity matrix of a sets of protein amino-acid
    # ===============================================================
    mlp_model = MLP(input_dim=24)
    mlp_model = torch.nn.DataParallel(mlp_model)
    mlp_ckpnt = torch.load(cfg.mlp_model_path)
    mlp_state_dict = mlp_ckpnt['model']
    mlp_model.load_state_dict(mlp_state_dict)
    mlp_model.cuda(0)

    mlp_precitor = MLP_Linkage_Predictor(mlp_model)
    mlp_precitor.predict_amino_linkage(protein_id="6XQ8")
