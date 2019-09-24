import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.init as init

class _GBN(nn.Module):
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'beta',
                     'running_mean', 'running_var', 'num_batches_tracked',
                     'num_features', 'affine']

    def __init__(self, opt, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_GBN, self).__init__()
        self.opt = opt
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.beta = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('beta', None)
        if self.track_running_stats:
            self.running_mean = Parameter(torch.Tensor(self.opt.micro_in_macro,1,num_features,1,1), requires_grad = False)
            self.running_var = Parameter(torch.Tensor(self.opt.micro_in_macro,1,num_features,1,1), requires_grad = False)
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.beta)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input):
        self._check_input_dim(input)
        N,C,_,_ = input.size()
        output,running_mean,running_var = self.g_b_n(input,self.running_mean,self.running_var,self.weight.expand(N,C),self.beta.expand(N,C))
        if self.training:
            self.running_mean.data = running_mean.data
            self.running_var.data = running_var.data
        return output

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(_GBN, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
    def g_b_n(self,input,running_mean,running_var,weight,beta):
        N,C,H,W = input.size()
        G = self.opt.micro_in_macro
        input = input.view(G,N//G,C,H,W)
        mean = torch.mean(input,(1,3,4),keepdim=True)
        var = torch.var(input,(1,3,4),keepdim=True)

        if self.training:
            running_mean = running_mean*self.momentum + mean*(1-self.momentum)
            running_var = running_var*self.momentum + var*(1-self.momentum)
            X_hat = (input-mean)/torch.sqrt(var+self.eps)
        else:
            X_hat = (input-running_mean)/torch.sqrt(running_var+self.eps)
        X_hat = X_hat.view(N,C,H,W)
        output = X_hat*weight.view(N,self.num_features,1,1)+beta.view(N,self.num_features,1,1)
        return output,running_mean,running_var


class GBN(_GBN):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class GCBN(_GBN):
    def __init__(self,opt,num_features):
        super(GCBN,self).__init__(opt,num_features)
        self.num_features = num_features
        inter_dim = 2*num_features
        self.gamma_mlp = nn.Sequential(
            nn.Linear(128,inter_dim),
            nn.ReLU(),
            nn.Linear(inter_dim,num_features)
        )
        self.beta_mlp = nn.Sequential(
            nn.Linear(128,inter_dim),
            nn.ReLU(),
            nn.Linear(inter_dim,num_features)
        )
    def forward(self,input,y):
        self._check_input_dim(input)
        N,C,_,_ = input.size()
        delta_gamma = self.gamma_mlp(y)
        delta_beta = self.beta_mlp(y)
        
        gamma_cloned = self.weight.clone()
        beta_cloned = self.beta.clone()
        gamma_cloned = gamma_cloned.view(1,C).expand(N,C)
        beta_cloned = beta_cloned.view(1,C).expand(N,C)

        gamma_cloned = gamma_cloned + delta_gamma
        beta_cloned = beta_cloned + delta_beta
        output,running_mean,running_var = self.g_b_n(input,self.running_mean,self.running_var,gamma_cloned,beta_cloned)
        if self.training:
            self.running_mean.data = running_mean.data
            self.running_var.data = running_var.data
        return output

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

