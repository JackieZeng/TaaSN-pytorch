# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support as prf
)

# -- accuracy, f1score, and confusion matrix --
def eval_correctness(seq_infos):
    # # point-based
    accp = accuracy_score(seq_infos['mode'], seq_infos['mode_pred'])
    f1scorep = prf(seq_infos['mode'], seq_infos['mode_pred'], average='macro')[2]
    confp = confusion_matrix(seq_infos['mode'], seq_infos['mode_pred'])
    
    # # distance-based
    accd = accuracy_score(seq_infos['mode'], seq_infos['mode_pred'], sample_weight=seq_infos.dl)
    f1scored = prf(seq_infos['mode'], seq_infos['mode_pred'], average='macro', sample_weight=seq_infos.dl)[2]
    confd = confusion_matrix(seq_infos['mode'], seq_infos['mode_pred'], sample_weight=seq_infos.dl)
    confd = np.around(confd/1000,0).astype(int)
    
    # # duration-based
    acct = accuracy_score(seq_infos['mode'], seq_infos['mode_pred'], sample_weight=seq_infos.dt)
    f1scoret = prf(seq_infos['mode'], seq_infos['mode_pred'], average='macro', sample_weight=seq_infos.dt)[2]
    conft = confusion_matrix(seq_infos['mode'], seq_infos['mode_pred'], sample_weight=seq_infos.dt)
    conft = np.around(conft/60, 0).astype(int)
    
    # --summarize--
    correctness = dict(
        accp=accp, f1scorep=f1scorep, confp=confp,
        accd=accd, f1scored=f1scored, confd=confd,
        acct=acct, f1scoret=f1scoret, conft=conft
    )
    return correctness


# -- trasfer --
def eval_transfer(seq_infos):
    M = np.unique(seq_infos['mode']).shape[0]
    ntraj = seq_infos.groupby('tid').size().shape[0]
    ntleg_true = 0
    ntleg_pred = 0
    ntrans_total = 0
    ae_transfer_total = 0
    # # --traverse trajectories--
    for tid,traj in seq_infos.groupby('tid'):
        # # --initialization--
        y_trip = traj['mode'].values
        y_pred = traj['mode_pred'].values
        trans_mat_true = np.zeros((M+1,M),dtype=int) # [start_trans|middle_trans]
        trans_mat_pred = np.zeros((M+1,M),dtype=int)
        # # --start transition--
        trans_mat_true[0,int(y_trip[0])] += 1
        trans_mat_pred[0,int(y_pred[0])] += 1
        # # --middle transition
        for i,j in zip(y_trip[:-1],y_trip[1:]):
            if i!=j:
                trans_mat_true[int(i)+1,int(j)] += 1
        for i,j in zip(y_pred[:-1],y_pred[1:]):
            if i!=j:
                trans_mat_pred[int(i)+1,int(j)] += 1
        # # --total absolute error--
        ntleg_true += np.sum(trans_mat_true)
        ntleg_pred += np.sum(trans_mat_pred)
        ntrans_total += np.sum(trans_mat_true)
        ae_transfer_traj = np.sum(np.abs(trans_mat_true - trans_mat_pred))
        ae_transfer_total += ae_transfer_traj
    mae_trasfer = ae_transfer_total / ntrans_total
    # --summarize--
    transfer = dict(
        ntraj=ntraj, ntleg_true=ntleg_true, ntleg_pred=ntleg_pred, mae_trasfer=mae_trasfer
    )
    return transfer
    


def evaluate_from_sequence(seq_infos):
    # 评价识别效果
    correctness = eval_correctness(seq_infos)
    transfer = eval_transfer(seq_infos)
    return correctness, transfer
        


def print_evaluation(correctness=None, transfer=None, logger=None, show_all=False):
    printf = logger.log_string if logger is not None else print

    # --correctness--
    if correctness is not None:
        # # point-based
        accp = correctness.get('accp')
        f1scorep = correctness.get('f1scorep')
        confp = correctness.get('confp')
        printf('--point--')
        printf('acc:\t**{:.2f}%**   f1-score:\t**{:.2f}%**'.format(100*accp,100*f1scorep), '\n', confp) 
        printf('Confusion Matrix - point')
        printf(np.around(confp / confp.sum(axis=1,keepdims=True) * 100,1))
        
        if show_all:
            # # distance-based
            accd = correctness.get('accd')
            f1scored = correctness.get('f1scored')
            confd = correctness.get('confd')
            printf('--distance--')
            printf('acc:\t**{:.2f}%**   f1-score:\t**{:.2f}%**'.format(100*accd,100*f1scored), '\n', confd) 
            printf('Confusion Matrix - distance')
            printf( np.around(confd / confd.sum(axis=1,keepdims=True) * 100,1) )
            
            # # duration-based
            acct = correctness.get('acct')
            f1scoret = correctness.get('f1scoret')
            conft = correctness.get('conft')
            printf('--duration--')
            printf('acc:\t**{:.2f}%**   f1-score:\t**{:.2f}%**'.format(100*acct,100*f1scoret), '\n', conft) 
            printf('Confusion Matrix - duration')
            printf( np.around(conft / conft.sum(axis=1,keepdims=True) * 100,1) )
        
        
    if transfer is not None:
        ntraj = transfer.get('ntraj')
        ntleg_true = transfer.get('ntleg_true')
        ntleg_pred = transfer.get('ntleg_pred')
        mae_trasfer = transfer.get('mae_trasfer')
        printf('N_trajectory: {:.0f}'.format(ntraj))
        printf('N_tripleg_true: {:.0f}'.format(ntleg_true))
        printf('N_tripleg_pred: {:.0f}'.format(ntleg_pred))
        printf('mae_transfer: {:.6f}'.format(mae_trasfer))


