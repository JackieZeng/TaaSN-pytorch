# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def readUserTraj(uid):
    # # 确定文件名
    traj_path = path_geolife + '{}/'.format(uid)
    traj_names = [tname for tname in os.listdir(traj_path) if '.csv' in tname]
    # # 读取每个轨迹
    trajs_user = []
    for tname in traj_names:
        traj_file = pd.read_csv(traj_path + tname)
        traj_file = traj_file.rename(columns={'latitude':'lat', 'longitude':'lng', 'mode':'modal'}) 
        trajs_user.append(traj_file)
    if len(trajs_user)==0:
        return []
    # # 合并所有轨迹
    trajs_user = pd.concat(trajs_user, ignore_index=True)
    trajs_user.loc[:,'lng'] = trajs_user.lng.astype(float)
    trajs_user.loc[:,'lat'] = trajs_user.lat.astype(float)
    trajs_user['modal'] = trajs_user.modal.map(mergeModal)
    # # 去掉北京之外的数据 东经115.7°—117.4°,北纬39.4°—41.6°
    trajs_user = trajs_user[(trajs_user.lng>115.7)&(trajs_user.lng<117.4)&
                            (trajs_user.lat>39.4)&(trajs_user.lat<41.6)]
    # # 把str格式的日期修改为timestamp
    trajs_user['timestamp'] = np.around(trajs_user.timestamp * 3600 * 24, 0).apply(lambda x:int(x))
    trajs_user = trajs_user.drop_duplicates(subset='timestamp',ignore_index=True) # 重复时刻记录只保留第一条
    trajs_user['uid'] = uid
    return trajs_user


def sphericalDistance(lng1, lat1, lng2, lat2): 
    '''经纬度可以是int、float，也可以是Series、ndarray、array'''
    lat1,lat2,lng1,lng2 = map(np.radians, [lat1,lat2,lng1,lng2])
    #AB两点的球面距离为:{arccos[sina*sinx+cosb*cosx*cos(b-y)]}*R  (a,b,x,y)  
    R = 6378.137 # 地球半径  km
    temp = np.sin(lat1)*np.sin(lat2)+\
            np.cos(lat1)*np.cos(lat2)*np.cos(lng2-lng1)  
    try:
        temp[temp>1.0] = 1.0
        d = np.arccos(temp)*R* 1000
        return np.around(d,3)
    except:
        temp = min(temp, 1.0)
        d = np.arccos(temp)*R* 1000
        return round(d, 3)


def calDistTo(data, by):
    lng_pre = data.groupby(by).lng.shift(1)
    lat_pre = data.groupby(by).lat.shift(1)
    dL = sphericalDistance(data.lng, data.lat, lng_pre, lat_pre)
    return dL


def calTimeTo(data, by):
    dt = data.timestamp - data.groupby(by).timestamp.shift(1)
    return dt


def trip_split_by_continuity(trajs_user, lbuffer=20, kvmax=2, tconti=200, ldwell=100, tdwell=1200):
    #
    def explode(dt,dL):
        v = dL/dt
        vexpec = (v.shift(2).fillna(30)+v.shift(1).fillna(30)+v.shift(-1).fillna(30)+v.shift(-2).fillna(30)) / 4
        vmax = kvmax * vexpec
        vmax[vmax<20] = 20
        abnormalspeed = (dL > vmax * dt + lbuffer)
        dwell = (dt>tdwell) & (dL<dt*(ldwell/tdwell))
        signalloss = dL > ((vmax * tconti + lbuffer - ldwell)/(tconti - tdwell)*(dt - tdwell) + ldwell)
        return (abnormalspeed | dwell | signalloss)
    
    # 格式处理
    trajs_user = trajs_user.reset_index(drop=True)
    
    # 找到间隔过大的点，打断
    dt_to = calTimeTo(trajs_user, by='uid')
    dL_to = calDistTo(trajs_user, by='uid')
    expl = explode(dt_to, dL_to)
    
    # tid
    trajs_user['tid'] = expl.cumsum()
    
    # 根据modal，赋予各方式段以tlegid
    is_change_modal = (trajs_user.modal != trajs_user.groupby('tid').modal.shift(1)) # 第一个值是True
    trajs_user['tlegid'] = is_change_modal.cumsum() - 1

    return trajs_user


def reassignid(trips):
    trips = trips.drop(columns=['dt_to','dL_to','v_to','dL_from','dt_from','v_from'], errors='ignore')
    is_tid_change = (trips.uid != trips.uid.shift(1)) | (trips.tid != trips.tid.shift(1))
    trips.loc[:,'tid'] = is_tid_change.cumsum() - 1
    is_tlegid_change = (trips.uid != trips.uid.shift(1)) | (trips.tlegid != trips.tlegid.shift(1))
    trips.loc[:,'tlegid'] = is_tlegid_change.cumsum() - 1
    return trips



#%% 为轨迹打标签，标上tid和tlegid
mergeModal = {0:0, 1:1, 2:2,
              3:3, 4:3,
              5:4, 6:4, 7:4}


trips = []

path_geolife = 'data/Geolife-corrected/'
uid_list = os.listdir(path_geolife)

for uid in tqdm(uid_list):
    # --读取该用户所有轨迹文件，合并到一个轨迹里，进行简单处理--
    trajs_user = readUserTraj(uid)
    if len(trajs_user) == 0:
        continue
    
    # --轨迹按时间、距离阈值切割--
    trajs_user = trip_split_by_continuity(trajs_user)
    trips.append(trajs_user)
    
    
trips = pd.concat(trips, ignore_index=True)
trips = reassignid(trips)
trips = trips.rename(columns={'modal':'mode'})

#
trips.to_csv('data/trips/trips.csv',index=False)

