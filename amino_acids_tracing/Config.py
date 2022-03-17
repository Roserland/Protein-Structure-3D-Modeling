class Config:
    def __init__(self) -> None:
        self.max_len = 512
        self.seq_loss_weight = 1.0
        self.pos_loss_weight = 1.0

        # Tensor board
        self.use_tb = False
        self.output_dir = './output_dir/'

        # Training paramters
        self.bacth_size = 4
        self.num_epochs = 10000
        self.lr = 0.001
        self.d_model = 512
        self.lr_mul = 0.99
        self.n_warmup_steps = 4000

        # label smoothing
        self.label_smoothing=False

        # model saving
        self.save_mode = "all"

    