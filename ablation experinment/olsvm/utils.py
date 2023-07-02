# -*- coding: utf-8 -*-
"""
Copyright (c) 2018-2019, SMaLL. All rights reserved.
@author:  Xingke Chen
@email:   chenxk1229@hotmail.com
@license: GPL-v3.
"""

import numpy as np

def log_nx(n, x):
    return np.log(x) / np.log(n)

def loss_parameter_updating(t, num_observations, num_classes):
    
        '''
        Adjust the ramp loss parameter adaptively.
        
        Parameters
        ----------
        t : integer, 
            Current time step
            
        num_observations : integer,
            Total number of observations
                           
        num_classes : integer, 
            Total number of categories  
            
        Returns
        -------
        The value of ramp loss parameter.
        
        return variable : number
            The ramp loss parameter after adaptively selecting
        '''
        
        assert 1 <= t <= num_observations,\
        "Error: current observation index is larger than the \
         number of observations"
        
        if 1 <= t < num_observations / 2:
            return -num_classes / 2 + num_classes / \
                   4 * log_nx(num_observations, num_observations / 2 - t)
        elif t == num_observations / 2:
            return -num_classes / 2
        else:
            return -num_classes / 2 - num_classes / \
                   4 * log_nx(num_observations, t - num_observations / 2)
        
