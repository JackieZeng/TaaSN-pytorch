# -*- coding: utf-8 -*-

from copy import deepcopy as dcp

# --Feature Normalization--
class StandardScalarDataFrame:
    def __init__(self, cols, eps=1e-10):
        self.cols = cols
        self.eps = eps
    
    def fit(self, X): # [nsample, nseq, nfeat, lenseq]
        X = X[self.cols]
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

    def transform(self, X):
        X = dcp(X)
        X_tra = (X[self.cols] - self.mean_) / (self.std_ + self.eps)
        X[self.cols] = X_tra[self.cols]
        return X
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        X = dcp(X)
        X_inv = (X[self.cols] * (self.std_ + self.eps)) + self.mean_
        X[self.cols] = X_inv[self.cols]
        return X




    