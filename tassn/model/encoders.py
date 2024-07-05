# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from copy import deepcopy as dcp


#%% --Auxiliary Modules--
def clones(module, N):
    return nn.ModuleList([dcp(module) for _ in range(N)])


#%% ==== Convolutional Encoder ====

class residualBlock(nn.Module):
    def __init__(self, dim, kernel_size=2):
        super().__init__()
        self.dim = dim
        # layers
        assert kernel_size % 2 == 1, f'kernel_size should be singular, but {kernel_size}'
        self.conv = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Conv1d(dim, dim * 2, kernel_size=kernel_size, padding='same'),
            nn.GLU(dim=-2),
            nn.Conv1d(dim, dim, kernel_size=1),
        )
        
    def forward(self, x):
        # x: [B,dim_in,T]
        x = x + self.conv(x) # -> [B,dim,T]
        return x # -> [B,dim,T]


class ConvEncoder(nn.Module):
    def __init__(self, C, M, num_conv, dim_conv, kernel_size, dim_out, pdrop=0.5):
        super().__init__()
        self.C = C # number of features
        self.M = M
        self.num_conv = num_conv
        self.dim_conv = dim_conv
        self.kernel_size = kernel_size # e.g., [2,3]
        self.dim_out = dim_out
        # layers
        self.projection = nn.Sequential(
            nn.Conv1d(C, dim_conv, kernel_size=1, padding='same'),
            nn.BatchNorm1d(dim_conv),
        )
        self.convs_res = clones(residualBlock(dim_conv, kernel_size), num_conv)
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(dim_conv),
            nn.Conv1d(dim_conv, dim_out, kernel_size=1),
        )
        self.fc_out = nn.Linear(dim_out, M)
        
        
    def get_device(self):
        return next(self.parameters()).device
    
    def feature_extraction(self, x):
        # x: [B,T,1+C]
        tw = x[..., [0]].detach() # [B,T,1]
        x = x[..., 1:].contiguous() # [B,T,C]
        # conv
        x = x.contiguous().permute(0,2,1) # -> [B,C,T]
        x = self.projection(x) # -> [B,dim_conv,T]
        for conv in self.convs_res:
            x = conv(x) # -> [B,dim_conv,T]
        x = self.mlp(x) # -> [B,dim_out,T]
        x = x.contiguous().permute(0,2,1) # -> [B,T,dim_out]
        # output
        x = torch.cat([tw, x], dim=-1) # -> [B,T,1+dim_out]
        return x # -> [B,T,dim_out]
        
    def forward(self, x):
        # x: [B,T,C]
        x = self.feature_extraction(x) # -> [B,T,1+dim_out]
        x = x[..., 1:] # [B,T,dim_out]
        return self.fc_out(x) # [B,T,M]
    
    
    
#%% ==== TW aggregation ====
class TimeWindowPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('zero', torch.FloatTensor([0]))
        self.layer_on_device = nn.ModuleList([nn.Linear(1, 1, bias=False)]) # for "getting device"
        
    def get_device(self):
        return next(self.parameters()).device
        
    def mw_pooling(self, tw, tw_uniq, x):
        # tw: [B,T,dim], tw_uniq: [B,Tu,1], x: [B,T,dim]
        B,T,dim = x.size()
        _,Tu,_ = tw_uniq.size()
        x = x.unsqueeze_(-2).expand(B,T,Tu,dim) # [B,T,dim] -> [B,T,1,dim] -> [B,T,Tu,dim]
        
        mask_sametw = ((tw >= tw_uniq.permute(0,2,1)) & (tw <= tw_uniq.permute(0,2,1)) ) # -> [B,T,Tu]
        mask_sametw = mask_sametw.unsqueeze(-1) # -> [B,T,Tu,1]
        
        # chunk to save memory if dim_hidden is too large
        n_parts = int(B * T * Tu * dim / (16 * 512 * 128 * 128)) + 1
        # print(f'{B} * {T} * {Tu} * {dim} / (16 * 512 * 128 * 128) + 1 = {n_parts}')
        # -- chunk & pooling --
        x_parts = x.chunk(n_parts, dim=-1)
        h_parts = []
        for x_p in x_parts:
            dim_p = x_p.size(-1)
            mask_p = mask_sametw.expand(B,T,Tu,dim_p) # [B,T,Tu,1] -> [B,T,Tu,dim]
            # -- pooling --
            h_p = torch.where(mask_p, x_p, self.zero) # -> [B,T,Tu,dim]
            h_parts.append(h_p.amax(dim=1)) # -> [B,Tu,dim]
            
        # concat
        hidden = torch.cat(h_parts, dim=-1).contiguous() # -> [B,Tu,dim]
            
        return hidden


    def forward(self, x, tw, mask):
        # x: [B,T,dim], tw: [B,T,1], mask: [B,T]
        tw[~mask] = torch.nan
        tw_max = np.nanmax(tw.cpu().numpy(), axis=1).squeeze(-1)
        tw_min = np.nanmin(tw.cpu().numpy(), axis=1).squeeze(-1)
        tw_uniq = pad_sequence(
            [torch.arange(tmi,tma + 1) for tmi,tma in zip(tw_min,tw_max)], 
            batch_first=True, padding_value=-99
        ).unsqueeze_(-1).to(self.get_device()) # -> [B,Tu,1], Tu: num of unique tw
        x = self.mw_pooling(tw, tw_uniq, x)
        
        return x # [B,Tu,dim*N]
        

class AggrEncoder(nn.Module):
    def __init__(self, dim_in, dim_out, M, pdrop=0.5):
        super().__init__()
        self.M = M
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.pdrop = pdrop
        # layers
        self.twpool = TimeWindowPooling()
        self.mlp_down = nn.Sequential(
            nn.BatchNorm1d(dim_in),
            nn.Conv1d(dim_in, dim_out, kernel_size=1),
        )
        self.fc_out = nn.Linear(dim_out, M)
        
    def get_device(self):
        return next(self.parameters()).device
    
    def feature_extraction(self, x, mask):
        # x: [B,T,1+dim_in], mask: [B,T]
        tw = x[..., [0]].detach() # [B,T,1]
        x = x[..., 1:].contiguous() # [B,T,dim_in]
        # pooling
        x = self.twpool(x, tw, mask) # -> [B,Tu,dim_in*N], Tu: num of unique tw
        x = x.permute(0,2,1)
        x = self.mlp_down(x)
        x = x.permute(0,2,1)
        return x # [B,T,dim_hidden]
        
    def forward(self,x, mask):
        # x: [B,T,C]
        x = self.feature_extraction(x, mask) # -> [B,Tu,dim_hidden]
        return self.fc_out(x) # [B,Tu,M]


#%% ==== Recurrent Encoder ====
class RecuEncoder(nn.Module):
    def __init__(self, dim_hidden, n_rnn, M, pdrop=0.3):
        super().__init__()
        assert dim_hidden % 2 == 0, f'{dim_hidden}(dim_hidden) % 2 != 0'
        self.dim_hidden = dim_hidden
        self.pdrop = pdrop
        # layers
        rnn_block = nn.LSTM(dim_hidden, int(dim_hidden/2), bidirectional=True, batch_first=True)
        self.rnn = clones(rnn_block, n_rnn)
        self.drop = clones(nn.Dropout(pdrop), n_rnn)
        self.fc_out = nn.Linear(dim_hidden, M)
        
    def get_device(self):
        return next(self.parameters()).device
    
    def feature_extraction(self, x):
        # shape of x: [B,T,dim_hidden]
        for rnn,drop in zip(self.rnn,self.drop):
            x0 = rnn(x)[0] # [B,T,dim_hidden] -> [B,T,dim_hidden]
            x = drop(x0 + x) # residual
        return x # [B,T,dim_hidden]

    def forward(self, x):
        x = self.feature_extraction(x) # [B,T,dim_hidden] -> [B,T,dim_hidden]
        return self.fc_out(x) # -> [B,T,M]
    
    
