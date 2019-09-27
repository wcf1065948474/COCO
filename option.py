class Option(object):
    def __init__(self):
        self.batchsize = 512
        self.latentsize = 100
        self.y_ebdsize = 28
        self.scale = 32
        self.latentoutsize = self.scale*16*2*2
        self.num_classes = 16
        self.micro_in_macro = 4
        self.macro_in_full = 4
        self.datadir='../input/celeba-dataset/img_align_celeba/img_align_celeba'
        self.datapath = '../input/celeba-h5py/celeba_img.h5py'
        self.macro_size = 32
        self.micro_size = 16
        self.full_size = 64
        self.LAMBDA = 10
        self.ALPHA = 100
        self.epoch = 50
        self.max_dataset = 50000
        self.my_model_dir = 'my_model'
        self.showgrad = False
        self.withspectral = False
        self.g_lr = 1e-4
        self.d_lr = 4e-4
        self.g_betas = (0.9,0.999)
        self.d_betas = (0.9,0.999)