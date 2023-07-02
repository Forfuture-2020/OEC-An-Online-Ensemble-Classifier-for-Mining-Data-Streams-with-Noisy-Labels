# -*- coding: utf-8 -*-
"""
Copyright (c) 2018-2019, SMaLL. All rights reserved.
@author:  Xingke Chen
@email:   chenxk1229@hotmail.com
@license: GPL-v3.
"""

import numpy as np
def shuffle(X, labels):
    '''
    Randomize the order of sample.

    Parameters
    ----------
    X : array-like, shape = [num_observations, num_features]
        Features of sample

    labels: array-like, shape = [num_observations]
        True labels for X

    Returns
    -------
    A tuple consists of X_shuffle and labels_shuffle

    X_shuffle : array, shape = [num_observations, num_features]
        The fueature matrix of sample after shuffling

    labels_shuffle : array shape = [num_observations]
        The true labels for X_shuffle
    '''

    randomize = np.arange(len(labels))
    # Create a random order
    np.random.shuffle(randomize)
    # Shuffle original dataset
    X_shuffle = X[randomize]
    labels_shuffle = labels[randomize]
    return X_shuffle, labels_shuffle 

def encoder(seq, unlabelled_mark = " "):
    '''
    Convert string-type label to general label 
    e.g. ['y','n','n','y', ' '] -> [0, 1, 1, 0, -1]
    
     Parameters
        ----------
        seq : array-like, with string elements
              the original labels.
              
        unlabelled_mark : the placeholder of the 
                          unlablled instance in
                          `seq`.
        
        Returns
        -------
        A tuple consists of encode_seq and encoder_dict
        
        encode_seq : array-like, with natural number
                     elements the label vector
                     after encoding.
                     
        encoder_dict : dictionary
                     the mapping from the original
                     strings to current numbers,
                     which will be ultilized
                     by decoder.
    '''
    valid_labels = set(seq).difference(unlabelled_mark)
    encoder_dict = dict(zip(valid_labels,range(len(valid_labels))))
    encoded_seq = np.zeros(len(seq), dtype = 'int32')
    for i in range(len(seq)):
        if seq[i] == unlabelled_mark:
            encoded_seq[i] = -1
        else:
            encoded_seq[i] = encoder_dict[seq[i]]
    return encoded_seq, encoder_dict   
        
        
def decoder(encoded_seq, encoder_dict):
    '''
    Convert encoded sequence to original sequence
    e.g. [0, 1, 1, 0, 1] -> ['y','n','n','y', 'y']
    
      Parameters
        ----------
        encoded_seq : array-like, with natural number elements
              the labels after encoding.
        
        encoder_dict : dictionary
              the mapping from the original strings to
              current numbers.
        
        Returns
        -------
        
        decode_seq : array-like, with string elements
                     the label vector after decoding,
                     that is, the original labels.
       
    
    '''
    decoder_dict = dict(zip(encoder_dict.values(),
                            encoder_dict.keys()))
    decode_seq = []
    for i in encoded_seq:
        decode_seq.append(decoder_dict[i])
    return decode_seq
