# -*- coding: utf-8 -*-

import pandas as pd

from ..preprocessing import dropout
from ..preprocessing import features
from . import metrics

class EvaluationPipeline:
    # 用于每个fold的模型测试
    def __init__(self, args, dropout_kwargs, scalar, model, batch_size):
        self.args = args
        self.dropout_kwargs = dropout_kwargs
        self.gap = dropout_kwargs['gap']
        self.scalar = scalar
        self.model = model
        self.batch_size = batch_size


    def sequence_identification(self, seq):
        # -- feature sequentialize --
        self.args.logger.log_string('calculating x and y...', end='')
        X_test_df, y_test_df = features.compute_xy(seq, self.args, sort=True)
        X_test_sc = self.scalar.transform(X_test_df)
        self.args.logger.log_string('done.')
        
        self.model.eval()
        y_test_df = y_test_df.set_index('itw', append=True)
        y_test_df = y_test_df.groupby(['uid','tid','itw'], sort=False).mode.max()
        
        # -- predict y --
        y_pred = self.model.predict(X_test_sc, self.batch_size, use_crf=True).cpu()
        y_pred_df = pd.DataFrame(y_pred, columns=['mode_pred'], index=y_test_df.index)
        
        # -- merge to the sequence --
        mode_of_points = seq[['uid','tid','ipnt','mode','dl','dt','timestamp','lng','lat']].copy()
        is_tlegid_change = (mode_of_points.uid!=mode_of_points.uid.shift(1)) | (mode_of_points['mode']!=mode_of_points['mode'].shift(1))
        mode_of_points['tlegid'] = is_tlegid_change.cumsum()-1
        
        X_test_df = self.scalar.inverse_transform(X_test_sc)
        seq_infos = X_test_df.reset_index()[['uid','tid','itw','ipnt']]
        seq_infos = seq_infos.merge(y_pred_df, on=['uid','tid','itw'])
        seq_infos = seq_infos.merge(mode_of_points , on=['uid','tid','ipnt'])
        seq_infos = seq_infos.set_index(['uid','tid','itw','ipnt'])
        return seq_infos


    def process(self, seq_raw):
        self.args.logger.log_string('*'*12, self.gap, '*'*12)
        
        # -- point dropout --
        self.seq_dpo = dropout.point_dropout(seq_raw, **self.dropout_kwargs)
        
        # -- TMI one point-level --
        self.seq_infos = self.sequence_identification(self.seq_dpo)
        
        # -- evaluate from sequence --
        correct, transfer= metrics.evaluate_from_sequence(self.seq_infos)
        metrics.print_evaluation(correct, transfer, logger=self.args.logger)
        return self.seq_infos