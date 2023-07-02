# -*- coding: utf-8 -*-
"""
Copyright (c) 2018-2019, SMaLL. All rights reserved.
@author:  Xingke Chen
@email:   chenxk1229@hotmail.com
@license: GPL-v3.
"""
import numpy as np
from abc import ABCMeta, abstractmethod
__all__ = ['Linear_Kernel', 'Polynomial_Kernel',
           'RBF_Kernel']
class Kernel(object):
    '''
    The abstract class for kernel
    '''
    @abstractmethod
    def compute_kernel(self, X_train, x_new):
        '''
        The abstract method is used to obtain the
        kernel matrix.
        '''
        raise NotImplementedError('This is an \
                                  abstract class')
    @abstractmethod
    def get_dimension(self):
        '''
        To obtain the scale of the problem, that is,
        the dimension of its feature space.
        '''
        raise NotImplementedError('This is an\
                                  abstract class')
        
class Linear_Kernel(Kernel):
    
    '''
    Linear kernel is used to compute the inner
    product of 2 vectors, namely, k(x, y) = <x, y>
    
    '''
    def __init__(self):
        self._dimension = None
        
    def compute_kernel(self, X_train, x_new):
        self._dimension = len(x_new)
        return np.dot(X_train, x_new) 
    
    def get_dimension(self):
        assert self._dimension != None, "Error: \
               No data input"
        return self._dimension
    
class Polynomial_Kernel(Kernel):
    
    '''
    Polynomial kernel is used to compute the
    polynomial transform of the inner product,
    namely, k(x, y) = (a<x, y> + b)^p
    where a is the scale factor
          b is the bias
          p is the degree of polynomial
    '''
    
    def __init__(self, scale_factor = 1, intercept = 1,
                 degree = 2):
        self._dimension = None
        self._scale_factor = scale_factor
        self._intercept = intercept
        self._degree = degree
        
    def compute_kernel(self, X_train, x_new):
        self._dimension = len(x_new) ** self._degree
        return (self._scale_factor * np.dot(X_train,
                x_new) + self._intercept)** self._degree
    
    def get_dimension(self):
        assert self._dimension != None, "Error:\
               No data input"
        return self._dimension 

class RBF_Kernel(Kernel):
    
    '''
    Radial Basis Function kernel is used to compute
    the Gaussian transform of the inner product,
    namely, k(x, y) = exp(-d||x - y||^2)
    where d is the precision parameter
    for Gaussian distribution.
    
    '''
    
    def __init__(self, d = 0.5):
        self._dimension = None
        self._precision = d

    def compute_kernel(self, X_train, x_new):
        shift = X_train - x_new
        if np.ndim(X_train) != 1:
            return np.exp(-self._precision * \
                          np.linalg.norm(shift, axis = 1)) 
        else:
            return np.exp(-self._precision * np.dot(shift, shift)) 
    def get_dimension(self):
        return np.inf
