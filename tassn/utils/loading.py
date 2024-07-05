# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import pandas as pd 


# --training and testing data loading--
def load_data(root_data, split, ifold, args):
    # reading data
    traj_seq = pd.read_csv(root_data + 'trips/trajseq.csv')
    
    # train-test-split
    assert split in ['trajsplit','usersplit']
    loc_path = root_data + 'train_test_split/{}/'.format(split)
    if split == 'trajsplit':
        file_test = 'tid_test_{}.csv'.format(ifold)
        tid_test = pd.read_csv(loc_path + file_test).values.flatten()
        mask_test = traj_seq.tid.isin(tid_test)
    else:
        file_test = 'uid_test_{}.csv'.format(ifold)
        uid_test = pd.read_csv(loc_path + file_test).values.flatten()
        mask_test = traj_seq.uid.isin(uid_test)
    
    seq_train = traj_seq[~mask_test]
    seq_test = traj_seq[mask_test]
    
    return seq_train, seq_test



# --Model Loading--
def load_model(path, truncate=False, freeze=True):
    net = torch.load(path)
    if truncate:
        net.fc_out = nn.Sequential()
    # print(net)
    for name, module in net._modules.items():
        # print(name)
        for param in module.parameters():
            if freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True
    return net
