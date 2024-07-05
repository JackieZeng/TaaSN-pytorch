# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .crf import CRF
from .encoders import ConvEncoder, AggrEncoder, RecuEncoder

from ..utils.dataset import Xdf_Set, collate_Xdf, BatchCutter


#%% ==== Full Model, consists of "Conv, Recu, CRF" ====
class FullModel(nn.Module):
    def __init__(self, 
                 C, M, 
                 num_conv, dim_conv, kernel_size, dim_out_cnn, pdrop_cnn,
                 dim_hidden, pdrop_agg,
                 n_rnn, pdrop_rnn
                 ):
        super().__init__()
        self.C = C
        self.M = M
        self.kernel_size = kernel_size
        self.dim_out_cnn = dim_out_cnn
        self.pdrop_cnn = pdrop_cnn
        self.dim_hidden = dim_hidden
        self.n_rnn = n_rnn
        self.pdrop_rnn = pdrop_rnn 
        self.batch_cutter = BatchCutter(nsmp_max=4*256, max_empty_ratio=0.3, B_min=1, sort=False)
        # layers
        self.cnn = ConvEncoder(C, M, num_conv, dim_conv, kernel_size, dim_out_cnn, pdrop_cnn)
        self.agg = AggrEncoder(dim_out_cnn, dim_hidden, M, pdrop_agg)
        self.rnn = RecuEncoder(dim_hidden, n_rnn, M, pdrop_rnn)
        self.crf = CRF(M, batch_first=True)
        
    
    def get_device(self):
        return next(self.parameters()).device
    
        
    def feature_extraction(self, x, mask=None, enc_name='rnn'):
        # shape of x: [B,T,1+C]([itw,feat]), mask: [B,T]
        assert enc_name in ['cnn','agg','rnn'], f'encoder should in [cnn,agg,rnn], but {enc_name}'
        x = self.cnn.feature_extraction(x) # -> [B,T,1+dim_hidden]
        if enc_name == 'cnn': return x
        else:
            assert mask is not None, 'Aggregation Module requires a mask as input'
            x = self.agg.feature_extraction(x, mask) # -> [B,Tu,dim_hidden]
            if enc_name == 'agg': return x
            else:
                x = self.rnn.feature_extraction(x) # -> [B,Tu,dim_hidden]
                return x
        
        
    def get_emissions(self, x, mask=None, enc_name='rnn'):
        # shape of x: [B,T,1+C], mask: [B,T]
        assert enc_name in ['cnn','agg','rnn'], f'encoder should in [cnn,agg,rnn], but {enc_name}'
        if enc_name == 'cnn': 
            return self.cnn.forward(x) # -> [B,T,M]
        else:
            assert mask is not None, 'Aggregation Module requires a mask as input'
            x = self.cnn.feature_extraction(x) # -> [B,T,dim_hidden]
            if enc_name == 'agg': 
                return self.agg.forward(x, mask) # -> [B,Tu,M]
            else:
                x = self.agg.feature_extraction(x, mask) # -> [B,Tu,M]
                return  self.rnn.forward(x) # -> [B,Tu,M]


    def calculate_loss(self, x, y, weight, mask, masktw, enc_name='rnn', reduction='token_mean'):
        # shape of x: [B,T,1+C], y: [B,T], mask: [B,T]
        x = self.get_emissions(x, mask, enc_name) # emissions, -> [B,Tu,M]
        return self.crf(x, y, weight, masktw, reduction)


    def forward(self, x, weight=None, mask=None, masktw=None, enc_name='rnn', use_crf=True):
        # shape of x: [B,T,1+C], mask: [B,T]
        assert enc_name in ['cnn','agg','rnn'], f'encoder should in [cnn,agg,rnn], but {enc_name}'
        x = self.get_emissions(x, mask, enc_name) # -> [B,Tu,M]
        if use_crf:
            return self.crf.decode(x, weight, masktw) # # list, [[y11, y12, ...], [y21, y22, ...], ...]
        else:
            return x # [B,T,M]
        
        
    def build_dataloader(self, X_df, batch_size):
        dataset = Xdf_Set( X_df )
        dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, collate_fn=collate_Xdf)
        return dataloader
    
    
    def predict(self, X_df, batch_size, use_crf=True):
        predloader = self.build_dataloader(X_df, batch_size)
        res_pred = []
        with torch.no_grad():
            for ibat, batch in enumerate(predloader):
                for batch_cut in self.batch_cutter(batch):
                    # get data
                    x_bat = batch_cut['X'].to(self.get_device()) # [B,T,C]
                    mask_bat = batch_cut['mask'].to(self.get_device()) # [B,T]
                    masktw_bat = batch_cut['masktw'].to(self.get_device()) # [B,Tu]
                    # predict
                    if use_crf:
                        w_bat = batch_cut['weight'].to(self.get_device()) # [B,Tu]
                        y_bat = self.forward(x_bat, weight=w_bat, mask=mask_bat, masktw=masktw_bat, enc_name='rnn', use_crf=True) # list, [[y11, y12, ...], [y21, y22, ...], ...]
                        y_bat = torch.LongTensor([yij for yi in y_bat for yij in yi]) # -> [ntw,]
                        res_pred.append( y_bat ) # [ntw,]
                    else:
                        oh_bat = self.forward(x_bat, mask=mask_bat, masktw=masktw_bat, enc_name='rnn', use_crf=False) # -> [B,T,M]
                        res_pred.append( oh_bat[masktw_bat].argmax(-1) ) # -> [ntw,]
        
        # return                 
        res_pred = torch.cat( res_pred ).cpu()
        return res_pred
            

