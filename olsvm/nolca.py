# -*- coding: utf-8 -*-

"""
Noise-resilient Large-scale Online Classification Algorithm.
Copyright (c) 2018-2019, SMaLL. All rights reserved.
@author:  Xingke Chen
@email:   chenxk1229@hotmail.com
@license: GPL-v3.
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from .kernel import Linear_Kernel, Polynomial_Kernel, RBF_Kernel
from .loss import Ramp
class NOLCA:
    
    '''A Noise-resilient Large-scale Online Classification
       Algorithm
    
       Implement a multiclass classification algorithm with
       noise-resilient loss function on the basis of online
       gradient descent[1]. Moreover, and it has a built-in
       module allowing binary semi-supervised classification.

       [1] Ling Jian, Fuhao Gao, Peng Ren, Yunquan Song,
           Shihua Luo, A Noise-Resilient Online Learning
           Algorithm for Scene Classification.
           Remote Sensing, 10(11),1836, 2018.
    
    '''
    
    def __init__(self, kernel = RBF_Kernel(),
                 loss = Ramp(),
                 num_support_vectors = 0,
                 support_vectors = [],
                 sample_weight =None):
        
        '''
        Initialize the parameters of the classifier,
        including the
        
        kernel type:
                                                self._kernel,
                                                
        loss type:
                                                  self._loss,
                                                  
        number of support vectors:
                                   self._num_support_vectors,
                                   
        collection of support vectors:
                                       self._support_vectors,
                                       
        collection of indices of the support vectors:
                                   self._support_vectors_idx,
        
        number of the wrong predictions:
                                             self._num_error,

        collection of accuracies of each round:
                                              self._accuracy,

        collection of predictions of each round:
                                       self._pred_collection,

        current round:
                                          self._current_step,
                                    
        Total number of observations:
                                      self._num_observations,
                                      
        The weight of samples:
                                      self._sample_weight,
        
        Parameters
        ----------
        kernel : Kernel
                 An instance of Kernel object, which can be
                 ``Linear_Kernel()``, ``Polynomial_Kernel()``
                 or ``RBF_Kernel()``.
                 Default value is the ``Linear_Kernel()``.

                 NOTE: Kernel instance can be created in
                       advance or created when invoking
                       the ``__init__`` function.
                         
                    
        loss :   Loss
                 An instance of Loss object. The candidates
                 canbe ``Hinge()`` for Hinge loss, or
                 ``Ramp()`` for Ramp loss.
                 Default value is ``Ramp()``.
        
        Examples
        --------
        >>> kernel = RBF_Kernel()
        >>> loss = Ramp(parameter = None, policy = "adaptive")
        >>> clf = NOLCA(kernel, loss)
            or
        >>> clf = NOLCA(RBF_Kernel(), Ramp())
        '''
        
        self._num_support_vectors = num_support_vectors
        if len(support_vectors) != 0:
            self._support_vectors = support_vectors
        else:
            self._support_vectors = []
        self._kernel = kernel
        self._loss = loss
        self._num_error = 0
        self._accuracy = []
        self._pred_collection = []
        self._num_observations = 0
        self._current_step = 0
        self._sample_weight = sample_weight
        self.window = 100
        
    def _predict(self, x):
        
        '''
        Obtain the prediction of the given observation x,
        which is given by current classifier
        (Internal function).
        
        Parameters
        ----------
        x : array-like, 
            the current observation
            
        Returns
        -------
        The prediction value of x.
        
        return variable : array-like
            
        '''
        if self._num_support_vectors >= self.window:
            return np.dot(self._sample_weight[-self.window:], x)
        else:
            return np.dot(self._sample_weight, x)
            
    def training(self, X, y, learning_rate = 0.01, reg_coefficient = 0.0, unlabelled = False):
        
        '''
        Train the classifier based on training dataset (X,y)

        Internal variables:
        

        The number of classes:
                                             self._num_classes,

        Confusion matrix:
                                        self._confusion_matrix,
                                        
        Parameters
        ----------
        X : array-like(generally 2D), 
            the observation n by p matrix, 
            
        y : array-like
            n by 1 label vector
            WARNING : the valid elements in y must be natural
                      numbers in order,for the string labels,
                      we provide encoder and decoder
                      functions. And if there are unlabelled
                      observations, the corresponding labels
                      should be -1.
            
        learning_rate : positive number, default 0.1
            the step size in online gradient
            descent algorithm

        reg_coefficient : non-negative scalar, default 0.0
            regularization coefficient
            
        unlabelled: boolean, default False
            indicate whether there are unlabelled observations in dataset, which is 
            used in semi-supervised binary classification setting. If there are full
            labels setting ``False`` and setting ``True`` otherwise.
            
        Returns
        -------
        No explicit return

        '''
        self._num_observations = len(y)
        
        labelset = set(y)

#         if -1 in labelset:
#             labelset.remove(-1)
        self._num_classes = len(labelset)

        self._confusion_matrix = np.zeros((self._num_classes,
                                           self._num_classes))
        for t in range(self._num_observations):
            
            self._update(X[t], y[t], learning_rate, reg_coefficient)
        
    def _update(self, x, y, learning_rate, reg_coefficient):
        
        '''
        1. Predict the label given observation x based on
           the previous classifier.
           
        2. Update the classifier by online gradient descent
           algorithm on the basis of current observation.
           
        (Internal function)
        Parameters
        ----------
        x : array-like, 
            the observation p by 1 vector, 
            
        y : array-like
            true label scalar

        learning_rate : positive number
            the step size in online gradient
            descent algorithm

        reg_coefficient : non-negative scalar
            regularization coefficient

        Returns
        -------
        No explicit return
            
        '''
        used_vector = []

        if 0 < self._num_support_vectors<self.window:

            kernel_vector = self._kernel.compute_kernel\
                            (np.array(self._support_vectors),x)
            
            pred_value = self._predict(kernel_vector)
        elif self._num_support_vectors >=self.window:
            kernel_vector = self._kernel.compute_kernel\
                            (np.array(self._support_vectors[-self.window:]),x)

            pred_value = self._predict(kernel_vector)
        # At the beginning of training, there are no
        # SVs, set all of the predictions to 0's. 
        else:
            pred_value = 1
#         print('pred_value',pred_value)
        
        if pred_value > 0:
            pred_label = 1
        else:
            pred_label = -1

        self._pred_collection.append(pred_label)
        true_label = y
#         if self._current_step % 1000 == 0:
#                 print(pred_value)
#                 print('y',y)
        # Compute the number of errors for labelled 
        # observations and error rate
        loss = None
#         if (true_label != -1):
#             self._count_num_error(pred_label, true_label)
#             loss = self._loss.loss_computing(pred_value[pred_label],
#                                              pred_value[true_label],
#                                              true_label, pred_label,
#                                              self._current_step, 
#                                              self._num_observations,
#                                              self._num_classes,
#                                              pred_value)
#             self._accuracy.append(self._get_accuracy(self._current_step + 1))  
#             self._current_step += 1
        
        # Update the classifier
        # self._count_num_error(pred_label, true_label)   ##shaokai：暂时先注释
        loss = self._loss.loss_computing(true_label, pred_label,
                                          self._current_step, 
                                          self._num_observations,
                                          self._num_classes,
                                          pred_value)
        self._accuracy.append(self._get_accuracy(self._current_step + 1))  
        self._current_step += 1
        gradient = None
#         if self._current_step % 1000 == 0:
#                 print('loss',loss)
#                 print('1-parameter', 1 - self._loss._parameter)
        if (0 < loss < 1 - self._loss._parameter) :
            if isinstance(self._sample_weight, np.ndarray):
                self._sample_weight = np.hstack((self._sample_weight,
                                                 np.zeros(1)))
            else:
                self._sample_weight = np.zeros(1)
            gradient = -true_label
#             if 0 < loss < 1 - self._loss._parameter:
#                 gradient = gradient_absolute
#             self._sample_weight[pred_label, self._num_support_vectors-1] = \
#                                             self._sample_weight[pred_label, self._num_support_vectors] - learning_rate * gradient

            self._sample_weight[self._num_support_vectors-1] = \
                                              -learning_rate * gradient
            if self._num_support_vectors > 1:
                self._sample_weight[:self._num_support_vectors-1] = \
                                             (1-learning_rate*reg_coefficient)*self._sample_weight[:self._num_support_vectors-1] 
            self._support_vectors.append(x)
            self._num_support_vectors += 1
        # Regularization
#         if self._num_support_vectors > 1 and reg_coefficient != 0:
#             self._sample_weight[:, :self._num_support_vectors - 1] = \
#                             (1 - learning_rate * reg_coefficient) * \
#             self._sample_weight[:, :self._num_support_vectors - 1]
#         if self._current_step % 5000 == 0:
#                 print('weight',self._sample_weight)
#                 print('gradient', gradient)
    def predicting(self, x):

        '''
        Predict the labels of new observations
        (without label).

        Parameters
        ----------
        x : array-like, 
            the observation p by 1 vector, 

        Returns
        -------

        np.argmax(pred_value) : integer,
            the prediction of current observation's
            label.
        
        '''
        kernel_vector = self._kernel.compute_kernel(np.array(self._support_vectors), x) 
        pred_value = self._predict(kernel_vector)
        return np.argmax(pred_value)
    def predicting_probal(self, x):

        '''
        Predict the labels of new observations
        (without label).

        Parameters
        ----------
        x : array-like, 
            the observation p by 1 vector, 

        Returns
        -------

        np.argmax(pred_value) : integer,
            the prediction of current observation's
            label.
        
        '''
        kernel_vector = self._kernel.compute_kernel(np.array(self._support_vectors), x) 
        pred_value = self._predict(kernel_vector)
        p1=np.exp(pred_value)/(1+np.exp(pred_value))
        p2=1-p1
        pred = {-1:p2,1:p1}
        return pred   
    def _count_num_error(self, pred_label, true_label):

        if pred_label != true_label:
            self._num_error += 1
            self._confusion_matrix[true_label, pred_label] += 1
        else:
            self._confusion_matrix[true_label, true_label] += 1
            
    def _get_accuracy(self, t):
        
        '''
        Compute the accuracy of the classifier at step t.
        (Internal function)
        
        '''
        return 1 - self._num_error / t
    
    def get_weight(self):
        
        '''
        Obtain the weight matrix of current classifier,
        which is useful when there is a necessity to
        store the classifier.
        
        '''
        
        return self._sample_weight
    
    def get_num_support_vectors(self):
        
        '''
        Obtain the number of support vectors.
        
        '''
        
        return self._num_support_vectors
    

    
    def get_support_vectors(self):

        '''
        Obtain the collection 
        of the support vectors.
        
        '''
        
        return self._support_vectors
    
    def get_num_error(self):

        '''
        Obtain total number of errors

        '''
        return self._num_error
    
    def get_accuracy(self):

        '''
        Obtain the collection of accuracies.
        
        '''
        
        return self._accuracy
    
    def get_confusion_matrix(self):

        '''
        Obtain confusion matrix.
        
        '''
        return self._confusion_matrix
    
    def get_prediction(self):
        
        '''
        Get all predictions so far.
        
        '''
        return np.array(self._pred_collection)

    def get_num_observations(self):
        
        '''
        Get total number of observations so far.
        
        '''
        return self._num_observations

    def get_num_classes(self):

        return self._num_classes
    
    def plot_accuracy_curve(self, x_label = "Number of samples",
                            y_label = "Accuracy"):

        '''
        Plot the accuracy curve, where x-axis represents the
        number of samples, y-axis represents the overall
        accuracy from observation 1 to current observation

        Parameters
        ----------
        x_label : string, 
            The name of x-axis.

        y_label = string,
            The name of y-axis.

        Returns
        -------

        The plot of accuracy curve.
            
        '''
        
        plt.plot(self._accuracy)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
    def plot_confusion_matrix(self, x_label = "True label",
                              y_label = "Prediction"):

        '''
        Plot the confusion matrix.

        Parameters
        ----------
        x_label : string, 
            The name of x-axis.

        y_label = string,
            The name of y-axis.

        Returns
        -------

        The plot of confusion matrix.
            
        '''
        
        plt.imshow(self._confusion_matrix)
        ticks = range(self._num_classes)
        plt.xticks(ticks)
        plt.yticks(ticks)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        loc = 0
        if np.any(self._confusion_matrix - 100 >= 0):
            loc = 0.25
        for x_tick in ticks:
            for y_tick in ticks:
                plt.text(x_tick-loc, y_tick, int(self._confusion_matrix[x_tick, y_tick]), color = "white")
        plt.show()
