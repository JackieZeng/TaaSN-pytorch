# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import time

from ..utils.dataset import Xydf_Set, collate_Xydf, BatchCutter
from . import optimization

from ..preprocessing import dropout
from ..preprocessing import features

#%%
class Trainer:
    def __init__(self, 
            args, 
            model,
            optimizer=None, criterion=nn.CrossEntropyLoss(), 
            scalar=None,
            start_epoch=0, 
            device='cpu', 
            verbose=False, 
        ):
        
        ### 初始化 encoding model ###
        self.model = model.to(device)
        
        ### 初始化 optimizer ###
        self.optimizer = (
            optimizer if optimizer is not None else 
            optim.Adam(self.model.parameters(), lr=args.lr)
        )
        
        ### 初始化 loss function ###
        self.criterion = criterion 
        
        ### Other Parameters ###
        # arguments
        self.args = args
        # data dropout
        self.dpo_kwargs = args.dpo_kwargs
        self.scalar = scalar
        # training hyper-parameter
        self.seed = args.seed
        self.start_epoch = start_epoch
        self.tol = args.tol
        self.lr = self.optimizer.param_groups[0]['lr'] if optimizer is not None else args.lr 
        # trainer setting
        self.logger = args.logger
        self.device = device
        self.verbose = verbose
        
        # optimizer
        self.schedule_kwargs = args.schedule_kwargs
        self.scheduler = optimization.get_scheduler(self.optimizer, **args.schedule_kwargs)
        # data loader
        self.batch_cutter = BatchCutter(**args.cutter_kwargs)
        # model arguments
        self.enc_stg1 = args.enc_stg1
        self.enc_stg2 = args.enc_stg2
        # training hyper-parameter
        self.B_stg1 = args.B_stg1
        self.B_stg2 = args.B_stg2
        self.epoch_stg1 = args.epoch_stg1
        self.epoch_stg2 = args.epoch_stg2
        self.max_epoch = self.epoch_stg1 + self.epoch_stg2
        
        
    def printf(self,*args,**kwargs):
        if self.verbose:
            if self.logger is not None:
                self.logger.log_string( *args,**kwargs )
            else:
                print(*args,**kwargs)
        
        
    ### ==== data preparation ====
    def calculate_xy(self, seq, seed):
        # point dropout
        self.printf('dropout points...', end='')
        seq_dpo = dropout.point_dropout(seq, mode='random', seed=seed,
            min_pnt = self.args.min_pnt, min_duration = self.args.min_duration,
            **self.args.dpo_kwargs
        )
        l1, l2 = len(seq_dpo), len(seq)
        self.printf('remain {:.2f}% ({}/{}), '.format(100*l1/l2, l1, l2), end='')
        # calculate x & y
        self.printf('calculating x and y...', end='')
        X_df, y_df = features.compute_xy(seq_dpo, self.args, sort=True)
        X_df = self.scalar.transform(X_df)
        self.printf('finish!')
        return X_df, y_df
    
    
    ### ==== build dataloader ====
    def build_trainloader(self, X_df, y_df, batch_size):
        self.printf('building training dataset (time window)...', end='')
        self.trainset = Xydf_Set( X_df, y_df)
        self.trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=collate_Xydf)
        self.printf('finish!')
        
        
    ### ==== start and finish training ====
    def lossbackpropa_stg1(self, batch):
        self.model.train()
        # --prediction and back-propa--
        nsmp = sum([len(m) for m in  batch['masktw']])
        loss = []
        for batch_cut in self.batch_cutter(batch):
            # X:[B,T,C+1], y:[B,Tu], mask:[B,T], masktw:[B,tw]
            X_cut = batch_cut['X'].to(self.device)
            y_cut = batch_cut['y'].to(self.device)
            mask_cut = batch_cut['mask'].to(self.device)
            masktw_cut = batch_cut['masktw'].to(self.device)
            
            # --prediction & loss calculation--
            emission = self.model.get_emissions(X_cut, mask_cut, enc_name=self.enc_stg1) # [B,T,C] => [B,Tu,M]
            loss_cut = self.criterion(emission[masktw_cut], y_cut[masktw_cut])
            del emission
            loss_cut = loss_cut.sum() / nsmp # reduction = tw_mean
            
            # --back propagation--
            loss_cut.backward()
            loss.append( loss_cut.detach() )
            del loss_cut
            
        loss = sum(loss).item()
        # --optimization--
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss, nsmp
    
        
    def lossbackpropa_stg2(self, batch):
        self.model.train()
        
        # --prediction and back-propa--
        nsmp = sum([len(m) for m in  batch['masktw']])
        loss = []
        for batch_cut in self.batch_cutter(batch):
            # X:[B,T,C+1], y:[B,Tu], mask:[B,T], masktw:[B,tw]
            X_cut = batch_cut['X'].to(self.device)
            y_cut = batch_cut['y'].to(self.device)
            mask_cut = batch_cut['mask'].to(self.device)
            masktw_cut = batch_cut['masktw'].to(self.device)
            w_cut = batch_cut['weight'].to(self.device)
            
            # --prediction & loss calculation--
            loss_cut = self.model.calculate_loss(X_cut, y_cut, w_cut, mask_cut, masktw_cut, enc_name=self.enc_stg2, reduction='none')
            loss_cut = loss_cut.sum() / nsmp
            
            # --back propagation--
            loss_cut.backward()
            loss.append( loss_cut.detach() )
            
        loss = sum(loss).item()
        # --optimization--
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss, nsmp
    
    
    def run_epoch(self, epoch, seq_train):
        "Standard Training and Logging Function"
        # point dropout, xy calculation
        X_df, y_df = self.calculate_xy(seq_train, seed=epoch)
        # initial loss
        loss_total = 0
        nsmp_total = 0
        
        # start
        if epoch < self.epoch_stg1:
            self.build_trainloader(X_df, y_df, self.B_stg1)
            # 前几个epoch，用预训练初始化rnn
            for ibat,batch in enumerate(self.trainloader):
                # --prediction, loss calculation, and back propagation--
                loss_pred_bat, nsmp = self.lossbackpropa_stg1(batch)
                # --record loss--
                loss_total += loss_pred_bat * nsmp
                nsmp_total += nsmp
        else:
            if epoch == self.epoch_stg1:
                self.best_loss = np.inf
                self.scheduler = optimization.get_scheduler(self.optimizer, **self.args.schedule_kwargs)
                    
            self.build_trainloader(X_df, y_df, self.B_stg2)
            # start
            for ibat,batch in enumerate(self.trainloader):
                # --prediction, loss calculation, and back propagation--
                loss_pred_bat, nsmp = self.lossbackpropa_stg2(batch)
                # --record loss--
                loss_total += loss_pred_bat * nsmp
                nsmp_total += nsmp
        # -- finish --
        loss_mean = loss_total / nsmp_total
        return loss_mean


    def fit(self, seq_train):
        # initialize
        self.best_loss = np.inf
        self.t_start = time.time()
        self.elapsed = 0
        
        # training
        for epoch in range(self.start_epoch, self.max_epoch):
            # --run one epoch--
            loss_epoch = self.run_epoch(epoch, seq_train)
            
            # 输出结果
            lr = self.optimizer.param_groups[0]['lr']
            self.elapsed = time.time() - self.t_start
            sentence = '[Epoch {}]: [Loss = {:.8f}], lr = {:.6f}, Elapsed: {:.2f} s'.format(epoch, loss_epoch, lr, self.elapsed)
            self.printf(sentence)
                
            # update learning rate
            self.scheduler.step()



