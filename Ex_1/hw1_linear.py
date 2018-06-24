# -*- coding: utf-8 -*-
"""
Created on  

@author: fame
"""

import numpy as np 
import matplotlib.pyplot as plt

def predict(X,W,b):  
    """
    implement the function h(x, W, b) here  
    X: N-by-D array of training data 
    W: D dimensional array of weights
    b: scalar bias

    Should return a N dimensional array  
    """
    return sigmoid(np.dot(X,W) + b)
 
def sigmoid(a): 
    """
    implement the sigmoid here
    a: N dimensional numpy array

    Should return a N dimensional array  
    """
    return (1.0/(1 + np.exp(-a)))

def l2loss(X,y,W,b):  
    """
    implement the L2 loss function
    X: N-by-D array of training data 
    y: N dimensional numpy array of labels
    W: D dimensional array of weights
    b: scalar bias

    Should return three variables: (i) the l2 loss: scalar, (ii) the gradient with respect to W, (iii) the gradient with respect to b
     """
    pred = predict(X, W, b)
    loss = (y - pred)
    pred = np.reshape(pred, (-1, 1))
    loss = np.reshape(loss, (-1, 1))
    sum_sqr_loss = np.sum(loss ** 2)
    
    dw = -2.0 * X * loss * pred * (1 - pred)
    db = -2.0 * loss * pred * (1 - pred)
    
    sum_dw = np.sum(dw, axis = 0) 
    sum_db = np.sum(db, axis = 0)
    return sum_sqr_loss, sum_dw, sum_db

def train(X,y,W,b, num_iters=1000, eta=0.001):  
    """
    implement the gradient descent here
    X: N-by-D array of training data 
    y: N dimensional numpy array of labels
    W: D dimensional array of weights
    b: scalar bias
    num_iters: (optional) number of steps to take when optimizing
    eta: (optional)  the stepsize for the gradient descent

    Should return the final values of W and b    
     """
     
    loss_values = []
    for i in np.arange(num_iters):
        loss, dw, db = l2loss(X, y, W, b)
        W += -eta * dw
        b += -eta * db
        loss_values.append(loss)
        
    x_val = np.arange(num_iters)
    plt.plot(x_val, loss_values)
    
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()    
    
    return W, b

 