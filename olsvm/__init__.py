# -*- coding: utf-8 -*- 
"""
Copyright (c) 2018-2019, SMaLL. All rights reserved.
@author:  Xingke Chen
@email:   chenxk1229@hotmail.com
@license: GPL-v3.
"""

from .nolca import NOLCA
from .kernel import Linear_Kernel
from .kernel import Polynomial_Kernel
from .kernel import RBF_Kernel
from .loss import Ramp
from .utils import log_nx
from .utils import loss_parameter_updating
from .preprocessing import shuffle
from .preprocessing import encoder
from .preprocessing import decoder


__all__ = [
    
     'NOLCA',
     'Linear_Kernel',
     'Polynomial_Kernel',
     'RBF_Kernel',
     'Ramp',
     'log_nx',
     'loss_parameter_updating',
     'shuffle',
     'encoder',
     'decoder',
     
]

