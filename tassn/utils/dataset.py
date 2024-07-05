# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np

#%% -- shortern --
def iter_data(traj, T):
    for idx,trip in traj.groupby(['uid','tid'], sort=False):
        if len(trip) <= T:
            yield trip
        else:
            while len(trip) > 0:
                itw_max = trip.itw.values[:T].max()
                mask = (trip.itw.values <= itw_max)
                yield trip.loc[mask]
                trip = trip.loc[~mask]
    
                
#%% -- Point Cloud DataSet --
class Xdf_Set(Dataset):
    def __init__(self, X_df, T_max=1e8):
        self.dtw = [xtrip.groupby(['itw'], sort=False).dtw.max().values for xtrip in iter_data(X_df, T_max)] # list of [Ti,]
        X_df = X_df.drop(columns=['dtw'])
        self.X = [xtrip for xtrip in iter_data(X_df, T_max)] # list of [Ti, C]
        
    def __getitem__(self, index):
        return {'X':self.X[index], 'dtw':self.dtw[index]}
 
    def __len__(self):
        return len(self.X)


def collate_Xdf(batch):
    # -- modify X --
    X_batch = [torch.FloatTensor(trip['X'].values) for trip in batch] # B * [Ti,C]
    # -- mask of X --
    mask_batch = [torch.ones(len(x)) == 1 for x in X_batch] # B * [Ti,]
    # -- mask of y --
    masktw_batch = [torch.ones(len(trip['X'].itw.unique())) == 1 for trip in batch] # B * [Tu,]
    # -- weight of emissions --
    weight_batch = [torch.tensor(trip['dtw']).float() for trip in batch] # B * [Tu,]
    # 将信息封装在字典res中
    res = dict(X = X_batch, mask = mask_batch, masktw = masktw_batch, weight=weight_batch)
    return res



class Xydf_Set(Dataset):
    def __init__(self, X_df, y_df, T_max=512):
        self.dtw = [xtrip.groupby(['itw'], sort=False).dtw.max().values for xtrip in iter_data(X_df, T_max)] # list of [Ti,]
        X_df = X_df.drop(columns=['dtw'])
        self.X = [xtrip.values for xtrip in iter_data(X_df, T_max)] # list of [Ti, C]
        self.y = [ytrip.groupby(['itw'], sort=False).mode.max().values for ytrip in iter_data(y_df, T_max)] # list of [Ti,]

    def __getitem__(self, index):
        return {'X':self.X[index], 'y':self.y[index], 'dtw':self.dtw[index]}
 
    def __len__(self):
        return len(self.X)


def collate_Xydf(batch):
    # -- modify X --
    X_batch = [torch.FloatTensor(trip['X']) for trip in batch] # B * [Ti,C]
    # -- mask of X --
    mask_batch = [torch.ones(len(x)) == 1 for x in X_batch] # B * [Ti,]
    # -- modify y --
    y_batch = [torch.LongTensor(trip['y']) for trip in batch] # B * [Tu,], Tu: num of unique tw
    # -- mask of y --
    masktw_batch = [torch.ones(len(y)) == 1 for y in y_batch] # B * [Tu,]
    # -- weight of emissions --
    weight_batch = [torch.tensor(trip['dtw']).float() for trip in batch] # B * [Tu,]
    # 将信息封装在字典res中
    res = dict(X = X_batch, y = y_batch, mask = mask_batch, masktw = masktw_batch, weight=weight_batch)
    return res



#%%
class BatchCutter:
    def __init__(self, nsmp_max, max_empty_ratio, B_min=2, sort=True):
        self.nsmp_max = nsmp_max
        self.max_empty_ratio = max_empty_ratio
        self.B_min = B_min
        self.sort = sort
    
    def create_batch_like(self, batch):
        return {kw:[] for kw in batch.keys()}
    
    def length_of(self, batch):
        for kw, matrix in batch.items():
            return len(matrix)
        
    def update_batch(self, batch_new, batch, idx):
        for kw,matrix in batch.items():
            batch_new[kw].append( matrix[idx] )
    
    def pack_batch(self, batch_new):
        padding_value_dict = {'X': 0, 'y': -1, 'mask': False, 'masktw': False, 'weight':1}
        for kw,matrix in batch_new.items():
            padding = padding_value_dict[kw]
            batch_new[kw] = pad_sequence(matrix, batch_first=True, padding_value=padding)
        return batch_new
    
    def should_be_cut_if_add_new_sample(self, i_sample, nsmp_total, empty_ratio):
        left_enough_sample = (i_sample + 1 <= self.B - self.B_min) if self.sort else True # sort=True happens in training, False in eval
        become_too_huge = (nsmp_total > self.nsmp_max)
        become_too_sparse = (empty_ratio >= self.max_empty_ratio)
        return (left_enough_sample and (become_too_huge or become_too_sparse))
            
    def is_last_sample(self, i_sample):
        return (i_sample == self.B - 1)        
    
    def __call__(self, batch):
        # measure
        npnts = np.array([len(m) for m in batch['mask']]) # B * [Ti,], num of points
        nsmps = np.array([len(m) for m in batch['masktw']]) # B * [Tui,], num of tw
        self.B = len(nsmps)
        self.T = nsmps.max()
        
        # sort
        if self.sort:
            idx_sort = np.argsort(nsmps)[::-1] # descending
            npnts_sort = npnts[idx_sort]
            nsmps_sort = nsmps[idx_sort]
        else:
            npnts_sort, nsmps_sort, idx_sort = npnts, nsmps, np.arange(self.B)
            
        # create
        batch_new = self.create_batch_like(batch)
        T_new = nsmp_valid = 0
        
        for i_sample, (Ti, Tui, idx) in enumerate(zip(npnts_sort, nsmps_sort, idx_sort)):
            if self.length_of(batch_new) >= self.B_min:
                # see what if add new trip
                B_new = self.length_of(batch_new) + 1
                nsmp_total = max(T_new, Tui) * B_new
                empty_ratio = 1 - (nsmp_valid + Tui) / nsmp_total # empty_ratio if add the new row
                
                if self.should_be_cut_if_add_new_sample(i_sample, nsmp_total, empty_ratio):
                    # pack the current 
                    batch_new = self.pack_batch(batch_new)
                    yield batch_new 
                    
                    # generate new one
                    batch_new = self.create_batch_like(batch)
                    T_new = nsmp_valid = 0
                
            # update the current
            self.update_batch(batch_new, batch, idx)
            T_new = max(T_new, Tui)
            nsmp_valid += Tui
                
            # if i_sample is the last one, then return no matter whether condition is fitted
            if self.is_last_sample(i_sample):
                batch_new = self.pack_batch(batch_new)
                yield batch_new
