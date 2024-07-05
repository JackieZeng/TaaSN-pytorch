# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 


class UniformSparsor:
    def __init__(self, gap):
        self.gap = gap

    def transform(self, traj):
        lt_gap = traj.timestamp.values // self.gap
        mask_kept = np.concatenate([[True], (lt_gap[1:] != lt_gap[:-1])])
        return mask_kept


class RandomSparsor:
    def __init__(self, dtmin, dtmax, lmd, ita):
        self.dtmin = dtmin
        self.dtmax = dtmax
        self.lmd = lmd
        self.ita = ita

    def set_seed(self, seed):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        return self
        
    def generate_gap(self, rng):
        p = rng.rand()
        if p < self.ita: # Exponential(lmd)
            gap = int(rng.exponential(scale=self.lmd))
            while gap < self.dtmin or gap > self.dtmax:
                gap = int(rng.exponential(scale=self.lmd))
        else: # Uniform(dtmin, dtmax)
            gap = int(rng.randint(low=self.dtmin, high=self.dtmax + 1))
        return gap
        
    def transform(self, traj):
        # check
        if len(traj) <= 2: return np.array([True] * len(traj))
        # 生成一个gap
        gap = self.generate_gap(self.rng)
        lt_gap = traj.timestamp.values // gap
        mask_kept = np.concatenate([[True], (lt_gap[1:] != lt_gap[:-1])])
        return mask_kept
    

def point_dropout(trips, mode, **kwargs):
    # sparsor initialization
    if mode == 'uniform':
        gap = kwargs.get('gap')
        sp = UniformSparsor(gap)
        
    elif mode == 'random':
        dtmin = kwargs.get('dtmin')
        dtmax = kwargs.get('dtmax')
        lmd = kwargs.get('lmd')
        ita = kwargs.get('ita')
        seed = kwargs.get('seed')
        sp = RandomSparsor(dtmin, dtmax, lmd, ita).set_seed(seed)
    
    # generate sparse trajectories trip-by-trip
    masks_dpo = []
    for tid,trip in trips.groupby('tid', sort=False):
        # generate a sparse trajectory
        mask_kept = sp.transform(trip)
        masks_dpo.append(mask_kept)
        
    # concat back
    masks_dpo = np.concatenate(masks_dpo)
    trips_dpo = trips[masks_dpo].copy()
        
    # remove too-short trips
    min_npnt = kwargs.get('min_pnt', 0)
    min_duration = kwargs.get('min_duration', 0)
    if min_npnt > 0 or min_duration > 0:
        npnt = trips_dpo.groupby('tid').size().rename('npnt')
        duration = (trips_dpo.groupby('tid').timestamp.max() - trips_dpo.groupby('tid').timestamp.min()).rename('duration')
        info = pd.concat([npnt,duration], axis=1)
        trips_dpo = trips_dpo.merge(info, on='tid', how='left')
        trips_dpo = trips_dpo[(trips_dpo.npnt >= min_npnt) & (trips_dpo.duration >= min_duration)]
        trips_dpo = trips_dpo.drop(columns=['npnt','duration'])
    
    return trips_dpo
