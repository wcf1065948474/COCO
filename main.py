import train
import dataset
import option
import torch
import models
import time
import utility
torch.backends.cudnn.benchmark = True

opt = option.Option()
celeba_dataset = dataset.CelebaDataset_h5py(opt)
dataloader = torch.utils.data.DataLoader(celeba_dataset,opt.batchsize,shuffle=True,num_workers=16,drop_last=True,pin_memory=True)
gan = train.COCOGAN(opt)

@utility.autotrain()
def train_net(gan,dataloader):
    for real_macro_list in dataloader:
        gan.latent_ebdy_generator.get_latent()
        for pos,real_macro in enumerate(real_macro_list):
            gan.train_parallel(real_macro,pos)
    gan.generate_parallel()
    gan.show_loss()

train_net(gan,dataloader)