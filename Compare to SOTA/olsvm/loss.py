# -*- coding: utf-8 -*- 
"""
Copyright (c) 2018-2019, SMaLL. All rights reserved.
@author:  Xingke Chen
@email:   chenxk1229@hotmail.com
@license: GPL-v3.
"""
import numpy as np
from .utils import loss_parameter_updating
from abc import ABCMeta, abstractmethod

__all__ = ['Ramp']
class Loss(object):
    '''
    The abstract class for loss function
    '''
    @abstractmethod    
    def loss_computing(self, x, y):
        raise NotImplementedError('This is an abstract method')
        
    @abstractmethod
    def get_parameter(self):
        raise NotImplementedError('This is an abstract method')
    
    
class Ramp(Loss):
    
    def __init__(self, parameter = -2, policy = "static"):
        '''
        Initialize the parameters of ramp loss.
        
        Parameters
        ----------
        parameter : number, default None when
                    using ``adaptive`` mode.
                    The ramp loss parameter
                    
        policy :    string, should be `adaptive` or `static`,
                    default `adaptive`
                    The strategy to choose ramp loss parameter,
                    `static` stands for fixing the parameter,
                    and `adaptive` means that parameter is chosen
                    according to data.
        
        Examples
        --------
        >>> loss = Ramp()
        or
        >>> loss = Ramp(parameter = None, policy = "adaptive")
        or
        >>> loss = Ramp(parameter = 1, policy = "static")
        '''
        assert (policy == "adaptive") or (policy == "static"),\
        "Error: the policy should be `adaptive` or `static`"
        self._policy = policy
        if self._policy == "adaptive":
            assert parameter == None,\
            "Error: do not assign any value to parameter in adaptive mode"
        else:
            assert (type(parameter) == int) or (type(parameter) == float),\
            "Error: parameter should be a number"
            self._parameter = parameter
        
    def loss_computing(self, true_label, pred_label,
                       t, num_observations, num_classes,pred_value):
        '''
        Compute the Ramp loss between the prediction
        of X_t and the label y_t.
        
        Parameters
        ----------
        pred_optimal :number
                      The prediction produced by the classifier
                      which achieves minimal loss
                      
        pred_true :   number
                      The prediction produced by the y_t-th classifier
                      
        pred_label :  number
                      The label given by the classifier
                      
        true_label :  number
                      The true label
        Returns
        -------
        return variable : number
                      Loss caused by X_t
        '''
        if pred_label == true_label:
            return 0
        else:
            if self._policy == "adaptive":
                self._parameter = loss_parameter_updating(t + 1,
                                                          num_observations,
                                                          num_classes)
            return  min(1 - self._parameter, max(0,
                    1 - true_label *pred_value))
    
    def get_parameter(self):
        '''
        To show the value of parameters.
        '''
        return self._parameter
