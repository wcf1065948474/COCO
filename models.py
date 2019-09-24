import torch
import torch.nn as nn
import numpy as np
import option
import time
import matplotlib.pyplot as plt
from GBN import GBN,GCBN


class GeneratorResidualBlock(nn.Module):
  def __init__(self,opt,input_channel,output_channel):
    super().__init__()
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.upscale = nn.utils.spectral_norm(nn.ConvTranspose2d(input_channel, input_channel, 4, 2, 1))
    self.upscale_branch = nn.utils.spectral_norm(nn.ConvTranspose2d(input_channel, input_channel, 4, 2, 1))
    self.conv1 = nn.utils.spectral_norm(nn.Conv2d(input_channel,output_channel,3,padding=1))
    self.conv2 = nn.utils.spectral_norm(nn.Conv2d(output_channel,output_channel,3,padding=1))
    self.conv_branch = nn.utils.spectral_norm(nn.Conv2d(input_channel,output_channel,3,padding=1))
    # self.cbn = ConditionalBatchNorm2d(output_channel)
    self.cbn = GCBN(opt,output_channel)

  def forward(self,input,y):
    master = self.relu1(input)
    master = self.upscale(master)
    master = self.conv1(master)
    master = self.cbn(master,y)
    master = self.relu2(master)
    master = self.conv2(master)
    branch = self.upscale_branch(input)
    branch = self.conv_branch(branch)
    return master+branch


class Generator(nn.Module):
  def __init__(self,opt):
    super().__init__()
    self.opt = opt
    self.linear = nn.Linear(opt.latentsize+opt.y_ebdsize,opt.latentoutsize)
    self.grb1 = GeneratorResidualBlock(opt,opt.scale*16,opt.scale*8)
    self.grb2 = GeneratorResidualBlock(opt,opt.scale*8,opt.scale*4)
    self.grb3 = GeneratorResidualBlock(opt,opt.scale*4,opt.scale*2)
    # self.grb4 = GeneratorResidualBlock(opt,64,32)
    self.model = nn.Sequential(
      # nn.BatchNorm2d(64),
      GBN(opt,opt.scale*2),
      nn.ReLU(),
      nn.utils.spectral_norm(nn.Conv2d(opt.scale*2,3,3,padding=1)),
      nn.Tanh()
    )
  def forward(self,input,y):
    res = self.linear(input)
    res = res.view(-1,opt.scale*16,2,2)
    res = self.grb1(res,y)
    res = self.grb2(res,y)
    res = self.grb3(res,y)
    # res = self.grb4(res,y)
    res = self.model(res)
    return res


class GeneratorResidualBlock_withoutspectral(nn.Module):
  def __init__(self,opt,input_channel,output_channel):
    super().__init__()
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.upscale = nn.ConvTranspose2d(input_channel, input_channel, 4, 2, 1)
    self.upscale_branch = nn.ConvTranspose2d(input_channel, input_channel, 4, 2, 1)
    self.conv1 = nn.Conv2d(input_channel,output_channel,3,padding=1)
    self.conv2 = nn.Conv2d(output_channel,output_channel,3,padding=1)
    self.conv_branch = nn.Conv2d(input_channel,output_channel,3,padding=1)
    # self.cbn = ConditionalBatchNorm2d(output_channel)
    self.cbn = GCBN(opt,output_channel)

  def forward(self,input,y):
    master = self.relu1(input)
    master = self.upscale(master)
    master = self.conv1(master)
    master = self.cbn(master,y)
    master = self.relu2(master)
    master = self.conv2(master)
    branch = self.upscale_branch(input)
    branch = self.conv_branch(branch)
    return master+branch


class Generator_withoutspectral(nn.Module):
  def __init__(self,opt):
    super().__init__()
    self.opt = opt
    self.linear = nn.Linear(opt.latentsize+opt.y_ebdsize,opt.latentoutsize)
    self.grb1 = GeneratorResidualBlock_withoutspectral(opt,opt.scale*16,opt.scale*8)
    self.grb2 = GeneratorResidualBlock_withoutspectral(opt,opt.scale*8,opt.scale*4)
    self.grb3 = GeneratorResidualBlock_withoutspectral(opt,opt.scale*4,opt.scale*2)
    # self.grb4 = GeneratorResidualBlock(opt,64,32)
    self.model = nn.Sequential(
      # nn.BatchNorm2d(64),
      GBN(opt,opt.scale*2),
      nn.ReLU(),
      nn.Conv2d(opt.scale*2,3,3,padding=1),
      nn.Tanh()
    )
  def forward(self,input,y):
    res = self.linear(input)
    res = res.view(-1,opt.scale*16,2,2)
    res = self.grb1(res,y)
    res = self.grb2(res,y)
    res = self.grb3(res,y)
    # res = self.grb4(res,y)
    res = self.model(res)
    return res


class DiscriminatorResidualBlock(nn.Module):
  def __init__(self,input_channel,output_channel,pooling=True):
    super().__init__()
    self.pooling = pooling
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.conv1 = nn.utils.spectral_norm(nn.Conv2d(input_channel,output_channel,3,padding=1))
    self.conv2 = nn.utils.spectral_norm(nn.Conv2d(output_channel,output_channel,3,padding=1))
    self.conv_branch = nn.utils.spectral_norm(nn.Conv2d(input_channel,output_channel,3,padding=1))
    if self.pooling == True:
      self.avg_pool = nn.AvgPool2d(2,2)
      self.avg_pool_branch = nn.AvgPool2d(2,2)

  def forward(self,input):
    master = self.relu1(input)
    master = self.conv1(master)
    master = self.relu2(master)
    master = self.conv2(master)
    if self.pooling == True:
      master = self.avg_pool(master)
      branch = self.avg_pool_branch(input)
      branch = self.conv_branch(branch)
    else:
      branch = self.conv_branch(input)
    return branch+master
    
class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.drb1 = DiscriminatorResidualBlock(3,opt.scale)
    self.drb2 = DiscriminatorResidualBlock(opt.scale,opt.scale*2)
    self.drb3 = DiscriminatorResidualBlock(opt.scale*2,opt.scale*4)
    self.drb4 = DiscriminatorResidualBlock(opt.scale*4,opt.scale*8)
    self.drb5 = DiscriminatorResidualBlock(opt.scale*8,opt.scale*8,False)
    self.relu = nn.ReLU()
    self.glb_pool = nn.AdaptiveMaxPool2d(1)
    self.linear = nn.Linear(opt.scale*8,1)
    self.linear_branch = nn.Linear(2,opt.scale*8)
    self.dah = nn.Sequential(
      nn.BatchNorm1d(opt.scale*8),
      nn.Linear(opt.scale*8,opt.scale*4),
      nn.BatchNorm1d(opt.scale*4),
      nn.LeakyReLU(),
      nn.Linear(opt.scale*4,2),#1->28
      nn.Tanh()
    )

  def forward(self,input,y):
    master = self.drb1(input)
    master = self.drb2(master)
    master = self.drb3(master)
    master = self.drb4(master)
    master = self.drb5(master)
    master = self.relu(master)
    master = self.glb_pool(master)
    master = torch.squeeze(master)
    h = self.dah(master)
    projection = self.linear_branch(y)
    projection = projection*master
    projection = torch.mean(projection,1,True)
    master = self.linear(master)
    return master+projection,h


