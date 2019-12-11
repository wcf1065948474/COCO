import models
import option
import torch
import os
import numpy as np
import pickle
from PIL import Image
import argparse


def rewrite_weight(epoch):
    weight = torch.load('{}_netG.pth'.format(epoch))
    tmp_w = torch.empty((16,1,64,1,1)).cuda()
    tmp_b = torch.empty((16,1,64,1,1)).cuda()
    tmp_w[0] = weight['model.0.weight'][0]
    tmp_w[2] = weight['model.0.weight'][0]
    tmp_w[8] = weight['model.0.weight'][0]
    tmp_w[10] = weight['model.0.weight'][0]
    tmp_w[1] = weight['model.0.weight'][1]
    tmp_w[3] = weight['model.0.weight'][1]
    tmp_w[9] = weight['model.0.weight'][1]
    tmp_w[11] = weight['model.0.weight'][1]
    tmp_w[4] = weight['model.0.weight'][2]
    tmp_w[6] = weight['model.0.weight'][2]
    tmp_w[12] = weight['model.0.weight'][2]
    tmp_w[14] = weight['model.0.weight'][2]
    tmp_w[5] = weight['model.0.weight'][3]
    tmp_w[7] = weight['model.0.weight'][3]
    tmp_w[13] = weight['model.0.weight'][3]
    tmp_w[15] = weight['model.0.weight'][3]

    tmp_b[0] = weight['model.0.bias'][0]
    tmp_b[2] = weight['model.0.bias'][0]
    tmp_b[8] = weight['model.0.bias'][0]
    tmp_b[10] = weight['model.0.bias'][0]
    tmp_b[1] = weight['model.0.bias'][1]
    tmp_b[3] = weight['model.0.bias'][1]
    tmp_b[9] = weight['model.0.bias'][1]
    tmp_b[11] = weight['model.0.bias'][1]
    tmp_b[4] = weight['model.0.bias'][2]
    tmp_b[6] = weight['model.0.bias'][2]
    tmp_b[12] = weight['model.0.bias'][2]
    tmp_b[14] = weight['model.0.bias'][2]
    tmp_b[5] = weight['model.0.bias'][3]
    tmp_b[7] = weight['model.0.bias'][3]
    tmp_b[13] = weight['model.0.bias'][3]
    tmp_b[15] = weight['model.0.bias'][3]

    weight['model.0.weight'] = tmp_w
    weight['model.0.bias'] = tmp_b

    torch.save(weight,'{}_netG.pth'.format(epoch))


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (0,2,3,1)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

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
        self.ebd = torch.transpose(self.ebd,0,1)
        self.ebd = np.repeat(self.ebd,self.opt.batchsize,0)
    def get_latent(self):
        # self.z = np.random.normal(0.0,1.0,(self.opt.batchsize,126)).astype(np.float32)
        self.z = np.random.uniform(-1.,1.,(self.opt.batchsize,126)).astype(np.float32)
        self.z = np.tile(self.z,(self.opt.num_classes,1))
        self.z = torch.from_numpy(self.z)
        self.latent = torch.cat((self.z,self.ebd),1)

class Evaluate(object):
    def __init__(self,opt):
        self.opt = opt
        self.wh = int(np.sqrt(opt.num_classes))
        self.G = models.Generator(opt)
        self.G.cuda()
        self.latent = Get_Latent_Y(opt)
        self.latent.get_latent()
        filename = "%s_netG.pth"%args.e
        self.G.load_state_dict(torch.load(filename))
        if args.eval:
            self.G.eval()

    def generate_imgs(self,args):
        with torch.no_grad():
            imgs = []
            start = time.time()
            for _ in range(args.times):
                macro_list = []
                self.latent.get_latent()
                for m in range(int(self.opt.num_classes/self.opt.N)):
                    tmp_latent = self.latent.latent[m*self.opt.N*self.opt.batchsize:(m+1)*self.opt.N*self.opt.batchsize]
                    tmp_latent = tmp_latent.cuda()
                    macro_list.append(self.G(tmp_latent,tmp_latent))
                if self.opt.N != 1:
                    micro_list = []
                    for macro in macro_list:
                        micro_list.extend(torch.chunk(macro,self.opt.N))
                else:
                    micro_list = macro_list
                w_imgs = []
                for w in range(self.wh):
                  w_imgs.append(torch.cat(micro_list[w*self.wh:(w+1)*self.wh],3))
                imgs.append(tensor2im(torch.cat(w_imgs,2)))
            print(time.time()-start)
        res = np.concatenate(imgs,0)
        res = res[:self.opt.max_dataset]
        with open('imgs','wb') as f:
            pickle.dump(res,f)
        
        count = 0
        if args.save_imgs:
            for i in range(args.save_imgs):
                img = Image.fromarray(res[count])
                img.save("gen_img//{}.jpg".format(count))
                count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='My COCO-GAN')
    parser.add_argument('-e',default=None)
    parser.add_argument('--eval',default=False,action='store_true')
    parser.add_argument('--times',default=792,type=int)
    parser.add_argument('--save_imgs',default=10)
    args = parser.parse_args()
    assert args.e is not None
    
    opt = option.Option()
    evaluator = Evaluate(opt)
    evaluator.generate_imgs(args)

    text = os.popen('nvidia-smi').readlines()
    for t in text:
      print(t,end='')




# import torch
# import torch.nn as nn
# from torch.nn.parameter import Parameter
# import torch.nn.init as init

# class _GBN(nn.Module):
#     __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
#                      'running_mean', 'running_var', 'num_batches_tracked',
#                      'num_features', 'affine']

#     def __init__(self, opt, num_features, eps=1e-5, momentum=0.01, affine=True,
#                  track_running_stats=True):
#         super(_GBN, self).__init__()
#         self.opt = opt
#         self.num_features = num_features
#         self.eps = eps
#         self.momentum = momentum
#         self.affine = affine
#         self.track_running_stats = track_running_stats
#         if self.affine:
#             self.weight = Parameter(torch.Tensor(16,1,num_features,1,1))
#             self.bias = Parameter(torch.Tensor(16,1,num_features,1,1))
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)
#         if self.track_running_stats:
#             self.running_mean = Parameter(torch.Tensor(self.opt.micro_in_macro,1,num_features,1,1), requires_grad = False)
#             self.running_var = Parameter(torch.Tensor(self.opt.micro_in_macro,1,num_features,1,1), requires_grad = False)
#         else:
#             self.register_parameter('running_mean', None)
#             self.register_parameter('running_var', None)
#         self.reset_parameters()
#         self.count = 0

#     def reset_running_stats(self):
#         if self.track_running_stats:
#             self.running_mean.zero_()
#             self.running_var.fill_(1)

#     def reset_parameters(self):
#         self.reset_running_stats()
#         if self.affine:
#             init.ones_(self.weight)
#             init.zeros_(self.bias)

#     def _check_input_dim(self, input):
#         raise NotImplementedError

#     def forward(self, input):
#         self._check_input_dim(input)
#         output = self.g_b_n(input,self.running_mean,self.running_var,self.weight[self.count:self.count+self.opt.N],self.bias[self.count:self.count+self.opt.N])
#         self.count += self.opt.N
#         self.count %= 16
#         return output

#     def extra_repr(self):
#         return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
#                'track_running_stats={track_running_stats}'.format(**self.__dict__)

#     def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
#                               missing_keys, unexpected_keys, error_msgs):
#         super(_GBN, self)._load_from_state_dict(
#             state_dict, prefix, local_metadata, strict,
#             missing_keys, unexpected_keys, error_msgs)
#     def g_b_n(self,input,running_mean,running_var,weight,beta):
#         N,C,H,W = input.size()
#         G = self.opt.N
#         input = input.view(G,N//G,C,H,W)
#         mean = torch.mean(input,(1,3,4),keepdim=True)
#         var = torch.var(input,(1,3,4),keepdim=True)
#         if self.training:
#             # running_mean.data = running_mean.data*(1-self.momentum) + mean*self.momentum
#             # running_var.data = running_var.data*(1-self.momentum) + var*self.momentum
#             X_hat = (input-mean)/torch.sqrt(var+self.eps)
#         else:
#             X_hat = (input-running_mean)/torch.sqrt(running_var+self.eps)
#         X_hat = X_hat*weight+beta
#         output = X_hat.view(N,C,H,W)
#         return output


# class GBN(_GBN):
#     def _check_input_dim(self, input):
#         if input.dim() != 4:
#             raise ValueError('expected 4D input (got {}D input)'
#                              .format(input.dim()))

# class GCBN(_GBN):
#     def __init__(self,opt,num_features):
#         super(GCBN,self).__init__(opt,num_features,affine=False)
#         self.num_features = num_features
#         self.G = self.opt.N
#         self.N = self.opt.batchsize
#         inter_dim = 2*num_features
#         self.gamma_mlp = nn.Sequential(
#             nn.Linear(128,inter_dim),
#             nn.ReLU(),
#             nn.Linear(inter_dim,num_features)
#         )
#         self.beta_mlp = nn.Sequential(
#             nn.Linear(128,inter_dim),
#             nn.ReLU(),
#             nn.Linear(inter_dim,num_features)
#         )
#     def forward(self,input,y):
#         self._check_input_dim(input)
#         delta_gamma = self.gamma_mlp(y)
#         delta_beta = self.beta_mlp(y)
#         delta_gamma = delta_gamma.view(self.G,self.N,self.num_features,1,1)
#         delta_beta = delta_beta.view(self.G,self.N,self.num_features,1,1)
#         output = self.g_b_n(input,self.running_mean,self.running_var,delta_gamma,delta_beta)
#         return output

#     def _check_input_dim(self, input):
#         if input.dim() != 4:
#             raise ValueError('expected 4D input (got {}D input)'
#                              .format(input.dim()))