
from .initial import create_path
import argparse

idx_cols = ['uid', 'tid']
ptord_cols = ['ipnt']
tw_cols = ['itw']
space_cols = ['lng','lat']
time_cols = ['timestamp']
label_cols = ['mode']
weight_cols = ['dtw']
feature_cols = ['dt','dl','distmin','nstops','distrd','distrl']


class Logger:
    def __init__(self, logpath, logname):
        self.logpath = logpath
        self.logname = logname
        create_path(logpath)
        self.log = open(logpath + logname, 'w')

    def log_string(self, *args, **kwargs):
        string = ' '.join([str(a) for a in args])
        end = kwargs.get('end','\n')
        self.log.write(string + end)
        self.log.flush()
        print(string, **kwargs)
        
    def close(self):
        self.log.close()


def parse_args(**kwargs):
    # obtain kwargs
    feat_cols = kwargs.pop('feat_cols', feature_cols)
    
    # create arguments
    parser = argparse.ArgumentParser()
    # logging
    logger = kwargs.pop('logger', None)
    if logger is None:
        logger = Logger('log/', 'tmp.txt')
    parser.add_argument('--logger', type=None, default=logger)
    
    # column arguments
    parser.add_argument('--idx_cols', type=list, default=idx_cols)
    parser.add_argument('--ptord_cols', type=list, default=ptord_cols)
    parser.add_argument('--tw_cols', type=list, default=tw_cols)
    parser.add_argument('--space_cols', type=list, default=space_cols)
    parser.add_argument('--time_cols', type=list, default=time_cols)
    parser.add_argument('--label_cols', type=list, default=label_cols)
    parser.add_argument('--weight_cols', type=list, default=weight_cols)
    parser.add_argument('--feat_cols', type=list, default=feat_cols, help = 'feature of X')
    
    # others
    kws = list(kwargs.keys())
    for kw in kws:
        parser.add_argument('--{}'.format(kw), type=None, default = kwargs.get(kw, None), help = ' ')
        
    # -- finish --    
    args = parser.parse_args()
    return args
