import math

class Config:
    def __init__(self) -> None:
        self.max_len = 512
        self.seq_loss_weight = 1.0
        self.pos_loss_weight = 1.0

        # Tensor board
        self.use_tb = False
        self.output_dir = './output_dir/'
        self.logs_dir = './output_dir/logs'

        # Training paramters
        self.bacth_size = 64
        self.num_epochs = 10000
        self.lr = 0.0001 * 2
        self.d_model = 512
        self.lr_mul = 0.99 * math.sqrt(self.bacth_size / 16)
        self.n_warmup_steps = 4000 / 4
        
        self.alpha = 0.5
        self.gamma = 2
        
        ## loss weight
        self.clf_loss_weight = 1.0
        self.pos_loss_weight = 1.0

        # label smoothing
        self.label_smoothing=False

        # model saving
        self.save_mode = "all"
        # self.best_model_path = './ckpnts_dir/2022-04-02-2145/model_accu_99.693.chkpt'
        self.best_model_path = './ckpnts_dir/2022-04-06-1700/model_accu_100.000.chkpt'
        self.linkage_model_path = './ckpnts_dir/2022-05-04/model_accu_99.996.chkpt'
        # self.linkage_model_path = './ckpnts_dir/2022-05-06-0000/model_accu_99.996.chkpt'

        self.model_output_dir = './ckpnts_dir/2022-04-06-1700/'

        # token settings
        self.pad_idx = 0.0
        self.bos_idx = 22.0
        self.eos_idx = 21.0
        ## decoder
        self.beam_size = 1
        

class ProteinProperties():
    """
        Some properties get by processing original pdb_files
    """
    def __init__(self) -> None:
        self.ca_dist_low = 2.0
        self.ca_dist_high = 4.2

        self.c_n_dist_low = 0.8
        self.c_n_dist_high = 1.2
    