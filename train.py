import torch
import torch.nn as nn
import option
import models
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (0,2,3,1)) + 1) / 2.0 * 255.0
    image_numpy = np.maximum(image_numpy, 0)
    image_numpy = np.minimum(image_numpy, 255)
    return image_numpy.astype(imtype)

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.show()

class Get_Latent_Y(object):
    def __init__(self,opt):
        self.opt = opt
        x = torch.linspace(-1,1,int(np.sqrt(opt.num_classes)))
        x = x.view(1,-1)
        x = x.expand(int(np.sqrt(opt.num_classes)),-1)
        y = torch.linspace(-1,1,int(np.sqrt(opt.num_classes)))
        y = y.view(-1,1)
        y = y.expand(-1,int(np.sqrt(opt.num_classes)))
        self.ebd = torch.stack((x,y),0)
        self.ebd = self.ebd.view(2,-1)
        self.wh = int(np.sqrt(opt.num_classes))
        self.pos_table = torch.arange(opt.num_classes).view(self.wh,self.wh)
        self.max_area = int(np.sqrt(opt.num_classes)-np.sqrt(opt.micro_in_macro)+1)
        self.macro_table = self.pos_table[0:self.max_area,0:self.max_area]
        self.macro_table = self.macro_table.contiguous().view(-1)

    def get_latent(self):
        # self.z = np.random.normal(0.0,1.0,(self.opt.batchsize,126)).astype(np.float32)
        self.z = np.random.uniform(-1.,1.,(self.opt.batchsize,126)).astype(np.float32)
        self.z = np.tile(self.z,(self.opt.micro_in_macro,1))
        self.z = torch.from_numpy(self.z)

    def get_ebdy(self,pos,mode='micro'):
        if mode == 'micro':
            pos_list = []
            for i in range(int(np.sqrt(self.opt.micro_in_macro))):
                for j in range(int(np.sqrt(self.opt.micro_in_macro))):
                    pos_list.append(self.macro_table[pos]+i*self.wh+j)
            ebdy = self.ebd[:,pos_list]
            ebdy = torch.transpose(ebdy,0,1)
        else:
            ebdy = self.ebd[:,self.macro_table[pos]].view(2,1)
            ebdy = torch.transpose(ebdy,0,1)
        ebdy = np.repeat(ebdy,self.opt.batchsize,0)
        return ebdy

    def get_latent_ebdy(self,pos):
        ebdy = self.get_ebdy(pos)
        return torch.cat((self.z,ebdy),1),ebdy

class COCOGAN(object):
    def __init__(self,opt):
        self.opt = opt
        self.G = models.Generator(opt) if opt.withspectral else models.Generator_withoutspectral(opt)
        self.D = models.Discriminator(opt)
        self.Lsloss = torch.nn.MSELoss()
        self.optimizerG = torch.optim.Adam(self.G.parameters(),opt.g_lr,opt.g_betas)
        self.optimizerD = torch.optim.Adam(self.D.parameters(),opt.d_lr,opt.d_betas)
        self.d_losses = []
        self.wd_losses = []
        self.g_losses = []
        self.wg_losses = []
        self.losses = {'d':[],'g':[],'wd':[],'wg':[]}
        self.G.cuda()
        self.D.cuda()
        self.G.apply(self.weights_init)
        self.D.apply(self.weights_init)
        self.latent_ebdy_generator = Get_Latent_Y(opt)

        self.generate_imgs_count = 0
    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            # if classname.find('Conditional') == -1:
            if m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def macro_from_micro_parallel(self,micro):
        macrolist = []
        hw = int(np.sqrt(self.opt.micro_in_macro))
        microlist = torch.chunk(micro,self.opt.micro_in_macro)
        for j in range(hw):
            macrolist.append(torch.cat(microlist[j*hw:j*hw+hw],3))
        return torch.cat(macrolist,2)

    def macro_from_micro_serial(self,micro):
        macrolist = []
        hw = int(np.sqrt(self.opt.micro_in_macro))
        for j in range(hw):
            macrolist.append(torch.cat(micro[j*hw:j*hw+hw],3))
        return torch.cat(macrolist,2)
    def calc_gradient_penalty(self,real_data,fake_data,ebd_y):
        alpha = torch.rand(self.opt.batchsize, 1,1,1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()
        ebd_y.requires_grad_()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates,_ = self.D(interpolates,ebd_y)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)

        gradient_penalty = (gradients[0].norm(2,dim=1)-1)**2
        return gradient_penalty.mean()*self.opt.LAMBDA

    def train_serial(self,x,pos):
        latent_ebdy,_ = self.latent_ebdy_generator.get_latent_ebdy(pos)
        micro_patches = []
        for k in range(self.opt.micro_in_macro):
            tmp_latent = latent_ebdy[k*self.opt.batchsize:k*self.opt.batchsize+self.opt.batchsize].cuda()
            micro_patches.append(self.G(tmp_latent,tmp_latent))
        self.macro_patches = self.macro_from_micro_serial(micro_patches)

        #update D()
        x = x.cuda()
        ebd_y = self.latent_ebdy_generator.get_ebdy(pos,'macro')
        ebd_y = ebd_y.cuda()
        self.D.zero_grad()
        self.macro_data = self.macro_patches.detach()
        fakeD,fakeDH = self.D(self.macro_data,ebd_y)#y有问题！
        realD,realDH = self.D(x,ebd_y)#y有问题！
        gradient_penalty = self.calc_gradient_penalty(x,self.macro_data,ebd_y)
        d_loss = fakeD.mean()-realD.mean()+gradient_penalty+self.opt.ALPHA*self.Lsloss(realDH,ebd_y)+self.opt.ALPHA*self.Lsloss(fakeDH,ebd_y)
        d_loss.backward()
        if self.opt.showgrad:
            plot_grad_flow(self.D.named_parameters())
        self.optimizerD.step()
        self.d_losses.append(d_loss.item())
        #update G()
        self.G.zero_grad()
        realG,realGH = self.D(self.macro_patches,ebd_y)#y有问题!
        g_loss = -realG.mean()+self.opt.ALPHA*self.Lsloss(realGH,ebd_y)
        g_loss.backward()
        if self.opt.showgrad:
            plot_grad_flow(self.G.named_parameters())
        self.optimizerG.step()
        self.g_losses.append(g_loss.item())
    def train_parallel(self,x,pos):
        latent_ebdy,_ = self.latent_ebdy_generator.get_latent_ebdy(pos)
        latent_ebdy = latent_ebdy.cuda()
        micro_patches = self.G(latent_ebdy,latent_ebdy)
        self.macro_patches = self.macro_from_micro_parallel(micro_patches)

        #update D()
        # x = x.cuda()
        ebd_y = self.latent_ebdy_generator.get_ebdy(pos,'macro')
        ebd_y = ebd_y.cuda()
        self.D.zero_grad()
        self.macro_data = self.macro_patches.detach()
        fakeD,fakeDH = self.D(self.macro_data,ebd_y)#y有问题！
        realD,realDH = self.D(x,ebd_y)#y有问题！
        gradient_penalty = self.calc_gradient_penalty(x,self.macro_data,ebd_y)
        wd_loss = fakeD.mean()-realD.mean()
        d_loss = wd_loss+gradient_penalty+self.opt.ALPHA*self.Lsloss(realDH,ebd_y)+self.opt.ALPHA*self.Lsloss(fakeDH,ebd_y)
        d_loss.backward()
        if self.opt.showgrad:
            plot_grad_flow(self.D.named_parameters())
        self.optimizerD.step()
        self.wd_losses.append(wd_loss.item())
        self.d_losses.append(d_loss.item())
        #update G()
        self.G.zero_grad()
        realG,realGH = self.D(self.macro_patches,ebd_y)#y有问题!
        wg_loss = -realG.mean()
        g_loss = wg_loss+self.opt.ALPHA*self.Lsloss(realGH,ebd_y)
        g_loss.backward()
        if self.opt.showgrad:
            plot_grad_flow(self.G.named_parameters())
        self.optimizerG.step()
        self.wg_losses.append(wg_loss.item())
        self.g_losses.append(g_loss.item())

    def generate_serial(self):
        # z = np.random.normal(0.0,1.0,(self.opt.batchsize,126)).astype(np.float32)
        z = np.random.uniform(-1.,1.,(self.opt.batchsize,126)).astype(np.float32)
        z = torch.from_numpy(z)
        ebdy = torch.transpose(self.latent_ebdy_generator.ebd,0,1)
        micro_list = []
        with torch.no_grad():
            for i in range(self.opt.num_classes):
                tmp_ebdy = ebdy[i].view(1,2)
                tmp_ebdy = tmp_ebdy.repeat(self.opt.batchsize,1)
                tmp_latent = torch.cat((z,tmp_ebdy),1).cuda()
                micro_list.append(self.G(tmp_latent,tmp_latent))
        hw = int(np.sqrt(self.opt.num_classes))
        hlist = []
        for i in range(hw):
            hlist.append(torch.cat(micro_list[i*hw:i*hw+hw],3))
        full_img = torch.cat(hlist,2)
        full_img = tensor2im(full_img)
        plt.figure(figsize=(2,2))
        plt.axis('off')
        plt.imshow(full_img)
        plt.show()
    
    def generate_parallel(self,save_imgs = False):
        pos_list = [0,2,6,8]
        self.latent_ebdy_generator.get_latent()
        macro_patches_list = []
        hw = int(np.sqrt(self.opt.macro_in_full))
        with torch.no_grad():
            for pos in pos_list:
                latent_ebdy,_ = self.latent_ebdy_generator.get_latent_ebdy(pos)
                latent_ebdy = latent_ebdy.cuda()
                micro_patches = self.G(latent_ebdy,latent_ebdy)
                macro_patches_list.append(self.macro_from_micro_parallel(micro_patches))
        tmp_list = []
        for i in range(hw):
            tmp_list.append(torch.cat(macro_patches_list[i*hw:i*hw+hw],3))
        full_img = torch.cat(tmp_list,2)
        full_img = tensor2im(full_img)
        if save_imgs:
            for i in range(self.opt.batchsize):
                self.generate_imgs_count += 1
                img = Image.fromarray(full_img[i])
                img.save("gen_img//{}.jpg".format(self.generate_imgs_count))
        else:
            hwlist = []
            plt.figure(figsize=(6,6))
            plt.axis('off')
            full_img = full_img[:16]
            imglist = np.split(full_img,full_img.shape[0])
            for i in range(4):
                hwlist.append(np.concatenate(imglist[i*4:i*4+4],2))
            full_img = np.concatenate(hwlist,1)
            plt.imshow(full_img[0])
            plt.show()

    

    def save_network(self,epoch_label):
        save_filename = "%s_netG.pth"%epoch_label
        save_path = os.path.join(self.opt.my_model_dir,save_filename)
        torch.save(self.G.state_dict(),save_path)
        save_filename = "%s_netD.pth"%epoch_label
        save_path = os.path.join(self.opt.my_model_dir,save_filename)
        torch.save(self.D.state_dict(),save_path)
        save_filename = "%s_losses"%epoch_label
        save_path = os.path.join(self.opt.my_model_dir,save_filename)
        with open(save_path,'wb') as f:
            pickle.dump(self.losses,f)

    def load_network(self,epoch_label):
        filename = "%s_netG.pth"%epoch_label
        filepath = os.path.join(self.opt.my_model_dir,filename)
        self.G.load_state_dict(torch.load(filepath))
        filename = "%s_netD.pth"%epoch_label
        filepath = os.path.join(self.opt.my_model_dir,filename)
        self.D.load_state_dict(torch.load(filepath))
        filename = "%s_losses"%epoch_label
        filepath = os.path.join(self.opt.my_model_dir,filename)
        with open(filepath,'rb') as f:
            self.losses = pickle.load(f)

    def update_learning_rate(self):
        for param_group in self.optimizerD.param_groups:
            param_group['lr'] = param_group['lr']*0.9
        for param_group in self.optimizerG.param_groups:
            param_group['lr'] = param_group['lr']*0.9
  
    def show_loss(self):
        avg_d_loss = np.mean(self.d_losses)
        avg_g_loss = np.mean(self.g_losses)
        avg_wd_loss = np.mean(self.wd_losses)
        avg_wg_loss = np.mean(self.wg_losses)
        print("d_loss={},g_loss={},wd_loss={},wg_loss={}".format(avg_d_loss,avg_g_loss,avg_wd_loss,avg_wg_loss))
        self.losses['d'].append(avg_d_loss)
        self.losses['g'].append(avg_g_loss)
        self.losses['wd'].append(avg_wd_loss)
        self.losses['wg'].append(avg_wg_loss)
        self.d_losses = []
        self.g_losses = []
        self.wd_losses = []
        self.wg_losses = []      
