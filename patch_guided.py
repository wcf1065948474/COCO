import torch
import torch.nn as nn
import random
import option
import models
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image


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

def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (0,2,3,1)) + 1) / 2.0 * 255.0
    image_numpy = np.maximum(image_numpy, 0)
    image_numpy = np.minimum(image_numpy, 255)
    return image_numpy.astype(imtype)



class PatchCOCOGAN(object):
    def __init__(self,opt):
        self.opt = opt
        self.G_img    = models.Generator(opt)
        self.D_img    = models.Discriminator(opt)
        self.G_latent = models.ImgToLatentGenerator(opt)
        self.D_latent = models.ImgToLatentDiscriminator()
        self.G_img_opt    = torch.optim.Adam(self.G_img.parameters(),opt.g_lr,opt.g_betas)
        self.D_img_opt    = torch.optim.Adam(self.D_img.parameters(),opt.d_lr,opt.d_betas)
        self.G_latent_opt = torch.optim.Adam(self.G_latent.parameters(),opt.g_lr,opt.g_betas)
        self.D_latent_opt = torch.optim.SGD(self.D_latent.parameters(),opt.g_lr)
        self.G_img.cuda()
        self.D_img.cuda()
        self.G_latent.cuda()
        self.D_latent.cuda()
        self.mseloss = nn.MSELoss()
        self.latent_ebdy_generator = Get_Latent_Y(opt)
    def macro_from_micro_parallel(self,micro):
        macrolist = []
        hw = int(np.sqrt(self.opt.micro_in_macro))
        microlist = torch.chunk(micro,self.opt.micro_in_macro)
        for j in range(hw):
            macrolist.append(torch.cat(microlist[j*hw:j*hw+hw],3))
        return torch.cat(macrolist,2)
    def calc_gradient_penalty(self,real_data,fake_data,ebd_y):
        alpha = torch.rand(self.opt.batchsize, 1,1,1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()
        # ebd_y.requires_grad_()
        differences = fake_data-real_data
        interpolates = real_data + (alpha * differences)
        interpolates = interpolates.cuda()
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates,_ = self.D_img(interpolates,ebd_y)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)
        gradient_penalty = (gradients[0].norm(2,dim=1)-1)**2
        return gradient_penalty.mean()*self.opt.LAMBDA
    def train(self,x,pos):
        latent_ebdy,_ = self.latent_ebdy_generator.get_latent_ebdy(pos)
        latent_ebdy = latent_ebdy.cuda()
        micro_patches = self.G_img(latent_ebdy,latent_ebdy)
        self.macro_patches = self.macro_from_micro_parallel(micro_patches)
        #update D_img()
        x = x.cuda()
        ebd_y = self.latent_ebdy_generator.get_ebdy(pos,'macro')
        ebd_y = ebd_y.cuda()
        self.D_img.zero_grad()
        self.macro_data = self.macro_patches.detach().clone()
        fakeD,fakeDH = self.D_img(self.macro_data,ebd_y)
        realD,realDH = self.D_img(x,ebd_y)
        gradient_penalty = self.calc_gradient_penalty(x,self.macro_data,ebd_y)
        wd_loss = fakeD.mean()-realD.mean()
        d_loss = wd_loss+gradient_penalty+self.opt.ALPHA*self.mseloss(realDH,ebd_y)+self.opt.ALPHA*self.mseloss(fakeDH,ebd_y)
        d_loss.backward()
        if self.opt.showgrad:
            plot_grad_flow(self.D_img.named_parameters())
        self.D_img_opt.step()
        # self.wd_losses.append(wd_loss.item())
        # self.d_losses.append(d_loss.item())
        #update G_latent()
        self.G_latent.zero_grad()
        predict_latent,predict_pos = self.G_latent(self.macro_data)
        g_latent_loss = self.mseloss(predict_latent,latent_ebdy[:self.opt.batchsize,:126])+self.mseloss(predict_pos,ebd_y)
        g_latent_loss.backward()
        self.G_latent_opt.step()
        #update G_img()
        self.G_img.zero_grad()
        realG,realGH = self.D_img(self.macro_patches,ebd_y)
        wg_loss = -realG.mean()
        g_loss = wg_loss+self.opt.ALPHA*self.mseloss(realGH,ebd_y)
        g_loss.backward()
        if self.opt.showgrad:
            plot_grad_flow(self.G_img.named_parameters())
        self.G_img_opt.step()
        # self.wg_losses.append(wg_loss.item())
        # self.g_losses.append(g_loss.item())
        #update D_latent()
        self.D_latent.zero_grad()
        fake_latent,fake_pos = self.G_latent(x)
        fake_latent_clone = fake_latent.detach().clone()
        fakeDLatent = self.D_latent(fake_latent_clone)
        realDLatent = self.D_latent(latent_ebdy[:self.opt.batchsize,:126])
        wdlatent_loss = fakeDLatent.mean()-realDLatent.mean()
        wdlatent_loss.backward()
        self.D_latent_opt.step()
        #update G_img()
        self.G_img.zero_grad()
        micro_ebd_y = self.latent_ebdy_generator.get_ebdy(pos)
        micro_ebd_y = micro_ebd_y.cuda()
        fake_latent_clone = fake_latent_clone.repeat((self.opt.micro_in_macro,1))
        fake_latent_cat = torch.cat((fake_latent_clone,micro_ebd_y),1)
        fakeimg = self.G_img(fake_latent_cat,fake_latent_cat)
        fakeimg = self.macro_from_micro_parallel(fakeimg)
        g_img_loss = self.mseloss(fakeimg,x)
        g_img_loss.backward()
        self.G_img_opt.step()
        #update G_latent()
        self.G_latent_opt.zero_grad()
        real_latent = self.D_latent(fake_latent)
        gl_loss = -real_latent.mean()+self.mseloss(fake_pos,ebd_y)
        gl_loss.backward()
        self.G_latent_opt.step()

    def predict(self,x):
        pos_list = [0,2,6,8]
        latent = self.G_latent(x)
        macro_patches_list = []
        hw = int(np.sqrt(self.opt.macro_in_full))
        with torch.no_grad():
            for pos in pos_list:
                micro_ebd_y = self.latent_ebdy_generator.get_ebdy(pos)
                micro_ebd_y = micro_ebd_y.cuda()
                latent_y = torch.cat((latent,micro_ebd_y),1)
                micro_patches = self.G_img(latent_y)
                macro_patches_list.append(self.macro_from_micro_parallel(micro_patches))
        tmp_list = []
        for i in range(hw):
            tmp_list.append(torch.cat(macro_patches_list[i*hw:i*hw+hw],3))
        full_img = torch.cat(tmp_list,2)
        full_img = tensor2im(full_img)

        for i in range(self.opt.batchsize):
            img = Image.fromarray(full_img[i])
            img.save("gen_img//{}.jpg".format(i))
                

