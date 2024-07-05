# -*- coding: utf-8 -*-

import torch.optim as optim


def get_optimizer(model, args, algo='adam'):
    params = [{'params': model.parameters(), "lr": args.lr}]
    if algo == 'adam':
        optimizer = optim.Adam(params)
    elif algo == 'sgd':
        optimizer = optim.SGD(params)
    else:
        raise ValueError('Undefined optimize algorithm:', algo)
    return optimizer



def get_scheduler(optimizer, **kwargs):
    policy = kwargs.get('policy', None)
    
    if policy == 'warmup':
        step_peak = kwargs.get('step_peak')
        step_half = kwargs.get('step_half')
        peak = kwargs.get('peak', 1)
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, 
            lr_lambda = lambda step: min(peak * (step + 1) / step_peak, peak * 2**(- ((step + 1) - step_peak) / step_half))
        )

    else:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1)

    return scheduler
        