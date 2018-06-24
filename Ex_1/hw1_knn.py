# -*- coding: utf-8 -*-
"""
Created on  

@author: fame
"""
 
import numpy as np 
 

def compute_euclidean_distances( X, Y ) :
    """
    Compute the Euclidean distance between two matricess X and Y  
    Input:
    X: N-by-D numpy array train
    Y: M-by-D numpy array test
    
    Should return dist: M-by-N numpy array   
    """  
     
    num_test = Y.shape[0]
    num_train = X.shape[0]
    dists = np.zeros((num_test, num_train))
    
    #||v-w||^2 = ||v||^2 + ||w||^2 - 2*dot(v,w)
    Y_square = np.reshape((Y**2).sum(axis=1), (-1, 1))
    X_square = np.reshape((X**2).sum(axis=1), (1, -1))
    
    dists = np.sqrt( Y_square + X_square - 2*(Y.dot(X.T)))
    return dists
 

def predict_labels( dists, labels, k=2):
    """
    Given a Euclidean distance matrix and associated training labels predict a label for each test point.
    Input:
    dists: M-by-N numpy array 
    labels: is a N dimensional numpy array
    
    Should return  pred_labels: M dimensional numpy array
    """
    num_test = dists.shape[0]
    pred_labels = np.zeros(num_test)
    
    # loop through all test data
    for i in xrange(num_test):
        # Calculating labels of k closest neighbours
        k_closest = labels[np.argsort(dists[i])][0:k]

        #Computing number of occrunce of each label
        counts = np.bincount(k_closest.astype(int))
        
        #predicting label with maximum occruence
        pred_labels[i] = np.argmax(counts)
        
    return pred_labels
         
         