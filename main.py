import train
import dataset
import option
import torch
import models
import time
import utility
torch.backends.cudnn.benchmark = True

def get_macro_from_full(img,pos):
    macro_pos_list = [0,1,2,4,5,6,8,9,10]
    macro_pos = macro_pos_list[pos]
    i = macro_pos//4
    j = macro_pos%4
    fake_patch = img[:,:,i*self.opt.micro_size:i*self.opt.micro_size+self.opt.macro_size,j*self.opt.micro_size:j*self.opt.micro_size+self.opt.macro_size]
    return fake_patch


opt = option.Option()
celeba_dataset = dataset.CelebaDataset_h5py(opt)
dataloader = torch.utils.data.DataLoader(celeba_dataset,opt.batchsize,shuffle=True,num_workers=16,drop_last=True,pin_memory=True)
gan = train.COCOGAN(opt)

@utility.autotrain()
def train_net(gan,dataloader):
    for real_macro_list in dataloader:
        real_macro_list = real_macro_list.cuda()
        gan.latent_ebdy_generator.get_latent()
        for pos in range(9):
            real_macro = get_macro_from_full(real_macro_list,pos)
            gan.train_parallel(real_macro,pos)
    gan.generate_parallel()
    gan.show_loss()

train_net(gan,dataloader)