# -*- coding: utf-8 -*-

from tassn.utils import loading
from tassn.utils import initial
from tassn.utils import config

from tassn.preprocessing import scalars, features, dropout
from tassn.model import classifiers
from tassn.training import optimization, trainers
from tassn.evaluation import metrics, eval_pipeline

import torch
import torch.nn as nn

import pandas as pd
import numpy as np
import datetime

# == Device of training ==
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# == hyper-parameters ==
dtmin = 2
dtmax = 330
lmd = 20
ita = 0.5
cols = ['dt','dl','v','a','w','distmin','nstops','distrd','distrl']
gap_test_list = [5,10,15,30,60,120,180,240,300]

split_list = ['usersplit','trajsplit']

#%% ==== Start hyper-parameter testing ====

for split in split_list:
    # -- initial parameters --
    # # set logging info
    log_path = 'log/'
    time_now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_name = '[Log-{}]-{}.txt'.format(time_now, split)
    logger = config.Logger(log_path, log_name)
    
    # # config hyperparameters
    args = config.parse_args(
        logger = logger, seed = 1,
        
        # -- column argumetns --
        feat_cols = cols,
        
        # -- data arguments --
        M = 5, C = len(cols),
        
        # -- model arguments --
        num_conv=6, dim_conv=128, kernel_size=3, dim_out_cnn=128, pdrop_cnn=0.5,
        dim_hidden = 128, pdrop_agg=0.5,
        n_rnn = 3, pdrop_rnn=0.3, 
        
        # -- dataloader arguments --
        cutter_kwargs = dict(nsmp_max=8*256, max_empty_ratio=0.6, B_min=3, sort=True), # prevent memory overflow, used in Trainer()
        
        # -- training arguments --
        algo = 'adam', 
        schedule_kwargs = dict(policy = 'warmup', step_peak = 2, step_half = 4, peak=1),
        epoch_stg1 = 15, epoch_stg2 = 15, 
        B_stg1 = 64, B_stg2 = 64, 
        enc_stg1 = 'agg', enc_stg2 = 'rnn',
        lr = 1e-3, tol = 1e-6,
        
        # -- point dropout arguments --
        dpo_kwargs = dict(dtmin = dtmin, dtmax = dtmax, lmd = lmd, ita = ita,), # point dropout training
        
        # -- trainer arguments --
        verbose = 1, 
        
        # -- tester arguments --
        min_pnt = 3, min_duration = 60,
        
        # -- file arguments --
        root_data = 'data/'
    )

    # # logging config
    args_str = ['Namespace(\n'] + ['\t' + ' = '.join(map(str, kv)) + ',\n' for kv in args._get_kwargs()] + [')']
    args_str = ''.join(args_str)
    args.logger.log_string(args_str)
    args.logger.log_string(f'======== [Case {split}] ========')

    # # record the predicted frames    
    seq_infos_dict = {gap:[] for gap in gap_test_list}
    
    # ==== Five-folds cross-validation ====
    for ifold in range(5):
        args.logger.log_string('------- [{} / Fold {}] -------'.format(split, ifold))
        
        # == loading trajectory-sequence ==
        seq_train, seq_test = loading.load_data(args.root_data, split, ifold, args)
        
        # == fit a `scalar` ==
        X_tmp = []
        for gap in gap_test_list:
            seq_tmp = dropout.point_dropout(seq_train, mode='uniform', gap=gap)
            X_tmp.append(features.compute_xy(seq_tmp, args)[0])
        X_tmp_df = pd.concat(X_tmp)
        
        scalar = scalars.StandardScalarDataFrame(cols=cols)
        scalar.fit(X_tmp_df)
        
        # == Model and Trainer initialization ==
        
        model_name = f'ce-rcrf_{ifold}'
        
        # -- set random seed --
        initial.set_random_state(args.seed)
        
        # -- model initialization --
        model = classifiers.FullModel(
            C=args.C, M=args.M, 
            num_conv=args.num_conv, dim_conv=args.dim_conv, kernel_size=args.kernel_size, 
            dim_out_cnn=args.dim_out_cnn, pdrop_cnn=args.pdrop_cnn,
            dim_hidden=args.dim_hidden, pdrop_agg=args.pdrop_agg,
            n_rnn=args.n_rnn, pdrop_rnn=args.pdrop_rnn, 
        )
        
        # -- initialize trainer of Transformer --
        optimizer = optimization.get_optimizer(model, args, algo=args.algo)
        criterion = nn.CrossEntropyLoss(reduction='none')

        # -- model training --
        trainer = trainers.Trainer(
            args, 
            model, 
            optimizer, criterion, 
            scalar = scalar,
            start_epoch = 0, 
            device = device, 
            verbose = args.verbose, 
        )

        # == Training and Saving model ==
        trainer.fit(seq_train)
        model = trainer.model
        
        
        # == Evaluating model ==
        for gap in gap_test_list:
            dropout_kwargs = dict(
                mode='uniform', gap = gap, min_pnt=args.min_pnt, min_duration=args.min_duration
            )
            
            pipe = eval_pipeline.EvaluationPipeline(args, dropout_kwargs, scalar, model, batch_size=128)
            seq_infos = pipe.process(seq_test)
            seq_infos_dict[gap].append(seq_infos)
            
            
    # ==== Evluating five-fold results ====
    seq_infos_all = {gap:pd.concat(seq_infos_cut) for gap, seq_infos_cut in seq_infos_dict.items()}
    correctness_all = {}
    transfer_all = {}
    for gap, seq_infos_cut in seq_infos_all.items():
        # 评价识别效果
        correctness, transfer = metrics.evaluate_from_sequence(seq_infos_cut)
        correctness_all[gap] = correctness
        transfer_all[gap] = transfer
        args.logger.log_string('#'*100)
        args.logger.log_string(f'**OVERALL** Validate with {split} & gap_{gap}')
        metrics.print_evaluation(correctness, transfer, logger=args.logger, show_all=True)
        args.logger.log_string('#'*100)
        
    #
    args.logger.close()

