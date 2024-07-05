# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 19:42:53 2022

@author: Jackie
"""
import numpy as np
import pandas as pd
from tassn.utils.initial import create_path
from tassn.preprocessing.features import sequence_transfer

# set column types
idx_cols = ['uid', 'tid', 'twid']
ptord_cols = ['ipnt']
space_cols = ['lat', 'lng']
time_cols = ['timestamp']
label_cols = ['mode']
feat_cols = ['dt', 'dl', 'distmin', 'nstops', 'distrd', 'distrl']
seq_cols = idx_cols + ptord_cols + space_cols + time_cols + label_cols + feat_cols

# load data
busstops = pd.read_csv('data/GIS/busstops.csv', encoding='gbk')
road = pd.read_csv('data/GIS/road_network_discrete_10m.csv', encoding='utf-8')
rail = pd.read_csv('data/GIS/rail_network_discrete_10m.csv', encoding='utf-8')

trips = pd.read_csv('data/trips/trips.csv')
trips = trips[['timestamp','lng','lat','uid','mode','tid','tlegid']]

print('calculate sequences of trajectories...')
traj_seq = sequence_transfer(trips, busstops, road, rail, sight_range=30, hop=60).reset_index(drop=True)

# select columns & save
traj_seq = traj_seq[seq_cols]
traj_seq.to_csv('data/trips/trajseq.csv', index=False)
  
      
# == leave trajectories out ==
num_folds = 5

# 按tid先后顺序划分
uidtid = traj_seq.groupby(['uid','tid']).size().rename('cnt')
uids = np.unique( traj_seq.uid )

save_path = 'data/train_test_split/trajsplit/'
create_path( save_path )
for ifold in range(5):
    tid_test = []
    for uid in uids:
        tids_user = list(uidtid.loc[uid].index)
        idxs = int( ifold * len(tids_user) / 5)
        idxe = int( (ifold+1) * len(tids_user) / 5)
        tid_test_user = tids_user[idxs:idxe]
        tid_test.extend(tid_test_user)
    tid_test = np.array(tid_test)
    info = pd.DataFrame(tid_test, columns=['tid'])
    info.to_csv(save_path + 'tid_test_{}.csv'.format(ifold), index=False)


# == leave travelers out ==
num_folds = 5

npnts = traj_seq.groupby(['uid']).size().sort_values(ascending=False)
folds = [[] for _ in range(num_folds)]

for i, (uid, _) in enumerate(npnts.items()):
    folds[int(i % num_folds)].append(uid) # 0 1 2 3 4 0 1 2 3 4


save_path = 'data/train_test_split/usersplit/'
create_path( save_path )
for ifold, uid_test in enumerate(folds):
    info = pd.DataFrame(uid_test, columns=['uid'])
    info.to_csv(save_path + 'uid_test_{}.csv'.format(ifold), index=False)
