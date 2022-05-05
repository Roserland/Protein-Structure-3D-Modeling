class Config:
    def __init__(self) -> None:
        self.max_len = 512
        self.seq_loss_weight = 1.0
        self.pos_loss_weight = 1.0

        # Tensor board
        self.use_tb = False
        self.output_dir = './output_dir/'

        # Training paramters
        self.bacth_size = 64
        self.num_epochs = 10000
        self.lr = 0.0001
        self.d_model = 512
        self.lr_mul = 0.99
        self.n_warmup_steps = 4000 / 2
        
        
        ## loss weight
        self.clf_loss_weight = 1.0
        self.pos_loss_weight = 1.0

        # label smoothing
        self.label_smoothing=False

        # model saving
        self.save_mode = "all"
        # self.best_model_path = './ckpnts_dir/2022-04-02-2145/model_accu_99.693.chkpt'
        self.best_model_path = './ckpnts_dir/2022-04-06-1700/model_accu_100.000.chkpt'
        self.linkage_model_path = './ckpnts_dir/2022-04-25-2100/model_accu_100.000_pos_los9.99722976402495e-07.chkpt'

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
    