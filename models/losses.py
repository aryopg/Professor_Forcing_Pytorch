import torch
import torch.nn as nn
from torch import autograd

from params import *

class JSDLoss(nn.Module):
    def __init__(self):
        super(JSDLoss,self).__init__()

    def forward(self, f_real, f_synt):
        assert f_real.size()[1] == f_synt.size()[1]

        f_num_features = f_real.size()[1]
        batch_size = f_real.size()[0]
        identity = autograd.Variable(torch.eye(f_num_features)*0.1)

        identity = identity.to(device)

        f_real_mean = torch.mean(f_real, 0, keepdim=True)
        f_synt_mean = torch.mean(f_synt, 0, keepdim=True)

        dev_f_real = f_real - f_real_mean.expand(batch_size,f_num_features) # batch_size x num_feat
        dev_f_synt = f_synt - f_synt_mean.expand(batch_size,f_num_features) # batch_size x num_feat

        f_real_xx = torch.mm(torch.t(dev_f_real), dev_f_real) # num_feat x batch_size * batch_size x num_feat = num_feat x num_feat
        f_synt_xx = torch.mm(torch.t(dev_f_synt), dev_f_synt) # num_feat x batch_size * batch_size x num_feat = num_feat x num_feat

        cov_mat_f_real = f_real_xx / (batch_size-1) - torch.mm(f_real_mean, torch.t(f_real_mean)) + identity # num_feat x num_feat
        cov_mat_f_synt = f_synt_xx / (batch_size-1) - torch.mm(f_synt_mean, torch.t(f_synt_mean)) + identity # num_feat x num_feat

        cov_mat_f_real_inv = torch.inverse(cov_mat_f_real)
        cov_mat_f_synt_inv = torch.inverse(cov_mat_f_synt)

        temp1 = torch.trace(torch.add(torch.mm(cov_mat_f_synt_inv, cov_mat_f_real), torch.mm(cov_mat_f_real_inv, cov_mat_f_synt)))
        temp2 = torch.mm(torch.mm((f_synt_mean - f_real_mean), (cov_mat_f_synt_inv + cov_mat_f_real_inv)), torch.t(f_synt_mean - f_real_mean))
        loss_g = temp1 + temp2

        return loss_g

class JSDLoss2(nn.Module):
    def __init__(self):
        super(JSDLoss2,self).__init__()

    def forward(self, f_real, f_synt):
        f_real_mean = torch.mean(f_real)
        f_synt_mean = torch.mean(f_synt)

        f_real_std = torch.std(f_real)
        f_synt_std = torch.std(f_synt)
        
        loss_p_q = self.custom_kl_div(f_real_mean, f_synt_mean, f_real_std, f_synt_std)
        loss_p_p = self.custom_kl_div(f_real_mean, f_real_mean, f_real_std, f_real_std)
        loss_q_p = self.custom_kl_div(f_synt_mean, f_real_mean, f_synt_std, f_real_std)
        loss_q_q = self.custom_kl_div(f_synt_mean, f_synt_mean, f_synt_std, f_synt_std)
        loss_g = (loss_p_q + loss_p_p + loss_q_p + loss_q_q) / 2

        return loss_g
    
    def custom_kl_div(self, mean1, mean2, std1, std2):
        part1 = torch.log(torch.div(std2, std1))
        part2 = torch.div(torch.pow(std1, 2) + torch.pow((mean1 - mean2), 2), 2*torch.pow(std2, 2))
        return part1 + part2 - 0.5
