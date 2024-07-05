# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 

from scipy.spatial import cKDTree # 投影坐标找k-最近点
from pyproj import Transformer
transformer = Transformer.from_crs('EPSG:4326', 'EPSG:4549')


def sphericalDistance(lng1, lat1, lng2, lat2): 
    '''经纬度可以是int、float，也可以是Series、ndarray、array'''
    lat1,lat2,lng1,lng2 = map(np.radians, [lat1,lat2,lng1,lng2])
    # AB两点的球面距离为:{arccos[sina*sinx+cosb*cosx*cos(b-y)]}*R  (a,b,x,y)  
    R = 6378.137 # 地球半径  km
    temp = np.sin(lat1)*np.sin(lat2)+\
            np.cos(lat1)*np.cos(lat2)*np.cos(lng2-lng1)  
    try:
        temp[temp>1.0] = 1.0
        d = np.arccos(temp)*R* 1000
        d[np.isnan(d)] = 0
        return np.around(d,3)
    except:
        temp = min(temp, 1.0)
        d = np.arccos(temp)*R* 1000
        return round(d, 3)


def calDistFrom(data, by):
    lng_nxt = data.groupby(by).lng.shift(-1)
    lat_nxt = data.groupby(by).lat.shift(-1)
    dl = sphericalDistance(data.lng, data.lat, lng_nxt, lat_nxt)
    return dl


def calTimeFrom(data, by):
    dt = data.groupby(by).timestamp.shift(-1) - data.timestamp
    return dt


def calDistxyFrom(data, by):
    lng_nxt = data.groupby(by).lng.shift(-1)
    lat_nxt = data.groupby(by).lat.shift(-1)
    
    dx_abs = sphericalDistance(data.lng, data.lat, lng_nxt, data.lat)
    dir_x = (data.lng <= lng_nxt).astype(int) * 2 - 1
    dy_abs = sphericalDistance(data.lng, data.lat, data.lng, lat_nxt)
    dir_y = (data.lat <= lat_nxt).astype(int) * 2 - 1
    
    return dir_x * dx_abs, dir_y * dy_abs


def calTrigonometric(data, by):
    dx, dy = calDistxyFrom(data, by)
    
    data['dx'], data['dy'] = dx, dy
    dx_nxt = data.groupby(by).dx.shift(-1)
    dy_nxt = data.groupby(by).dy.shift(-1)
    data.drop(columns=['dx', 'dy'], inplace=True)
    
    cos = (dx * dx_nxt + dy * dy_nxt) / ((dx**2 + dy**2) * (dx_nxt**2 + dy_nxt**2) + 1e-2)**0.5
    cos = cos.fillna(1)
    sin = (dx * dy_nxt - dx_nxt * dy) / ((dx**2 + dy**2) * (dx_nxt**2 + dy_nxt**2) + 1e-2)**0.5
    sin = sin.fillna(0)
    return cos, sin



def buildKDTree(busstops):
    # Prepare btree arrays
    y, x = transformer.transform(busstops.lat, busstops.lng)
    stops_xy = pd.DataFrame(np.array([x, y]).T, columns=['x','y'])
    # build a k-d tree for euclidean nearest node search
    btree = cKDTree(data=stops_xy, compact_nodes=True, balanced_tree=True)
    return btree


def get_k_nearest_points(trip, btree, k=50):
    # query the distances and ids of k-closest nodes to each point
    y, x = transformer.transform(trip.lat, trip.lng)
    points = pd.DataFrame(np.array([x,y]).T, columns=['x','y'])
    Dist, nIdxs = btree.query(points, k=k) # nIdxs是stop在stops_xy里的index
    Dist = Dist.reshape(-1, k)
    nIdxs = nIdxs.reshape(-1, k)
    Dist = pd.DataFrame(Dist,columns=range(Dist.shape[1]),index=trip.index)
    return Dist, nIdxs
    

def calNearestPointDistance(Dist):
    return Dist.min(axis=1)


def calNNearbyStops(Dist, sight_range):
    return (Dist<=sight_range).sum(axis=1)


def sequence_transfer(trips, busstops, road, rail, sight_range=30, hop=60):
    # preparation
    trips['n'] = 1
    trips['ipnt'] = trips.groupby('tid').n.cumsum() - 1
    trips = trips.drop(columns=['n'])
    
    # twid starts from 0
    trips['twid'] = (trips.timestamp // hop).astype(int)
    tw_min = (trips.groupby(['tid']).timestamp.min() // hop).astype(int).rename('tw_min')
    trips = trips.merge(tw_min, on=['tid'], how='left')
    trips['twid'] = (trips.twid - trips.tw_min).astype(int) # for index
    
    # motion feature preparation
    trips['dl'] = calDistFrom(trips, by='tid')
    trips['dt'] = calTimeFrom(trips, by='tid').fillna(1)
    
    # bus feature preparation
    btree = buildKDTree(busstops)
    Dist,_ = get_k_nearest_points(trips, btree, k=50)
    distmin = calNearestPointDistance(Dist).rename('distmin')
    nstops = calNNearbyStops(Dist, sight_range).rename('nstops')
    trips['distmin'] = 1 / (1 + distmin / 100)
    trips['nstops'] = 1 / (1 + nstops / 10)
    
    # road network feature preparation
    btree_rd = buildKDTree(road)
    Dist_rd,_ = get_k_nearest_points(trips, btree_rd, k=1)
    distrd = calNearestPointDistance(Dist_rd).rename('distrd')
    trips['distrd'] = 1 / (1 + distrd / 20)
    
    # road network feature preparation
    btree_rl = buildKDTree(rail)
    Dist_rl,_ = get_k_nearest_points(trips, btree_rl, k=1)
    distrl = calNearestPointDistance(Dist_rl).rename('distrl')
    trips['distrl'] = 1 / (1 + distrl / 50)
    
    return trips 


def seqence_sorting(seq_df, index, columns):
    def sort_tid_by_npnt(X_df):
        info_df = X_df.iloc[:,[0]].dropna().reset_index()
        npnt_sort = info_df.groupby(['tid']).size().rename('npnt').reset_index().sort_values(['npnt','tid']).set_index(['tid'])
        return npnt_sort.index
    # data sorting by number of tw
    index_sort = sort_tid_by_npnt(seq_df)
    seq_df = (
        seq_df.reset_index().set_index('tid')
        .loc[index_sort].reset_index()
        .set_index(index)[columns]
    )
    # finish
    return seq_df


def compute_xy(trips, args, sort=False):
    #
    trips['dl'] = calDistFrom(trips, by='tid')
    trips['dt'] = calTimeFrom(trips, by='tid').fillna(1)
    trips['v'] = (trips.dl / trips.dt).fillna(0)
    dv = (trips.groupby('tid').v.shift(-1) - trips.v).fillna(0)
    trips['a'] = (dv / trips.dt).fillna(0)
    cos, sin = calTrigonometric(trips, by='tid')
    trips['w'] = (np.arccos(cos) / trips.dt).fillna(0)
    
    #
    trips['dtw'] = (trips.groupby('tid').twid.shift(-1) - trips.twid).fillna(1)
    
    #
    is_twid_change = ((trips.tid != trips.tid.shift(1)) | (trips.twid != trips.twid.shift(1)))
    trips['itw'] = is_twid_change.cumsum() - 1
    itw_min = trips.groupby(['tid']).itw.min().rename('itw_min')
    trips = trips.merge(itw_min, on=['tid'], how='left')
    trips.loc[:,'itw'] = (trips.itw - trips.itw_min).astype(int) # for index
    trips.drop(columns=['itw_min'], inplace=True)
    
    # calculate X and y
    idx_xy = (args.idx_cols + args.ptord_cols)
    cols_x = (args.tw_cols + args.feat_cols + args.weight_cols)
    cols_y = (args.tw_cols + args.label_cols)
    X_df = trips[idx_xy + cols_x].set_index(idx_xy)
    y_df = trips[idx_xy + cols_y].set_index(idx_xy)
    if sort:
        X_df = seqence_sorting(X_df, index=idx_xy, columns=cols_x)
        y_df = seqence_sorting(y_df, index=idx_xy, columns=cols_y)
    # finish
    return X_df, y_df