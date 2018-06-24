# -*- coding: utf-8 -*-
"""
Created on 

@author: fame
"""

 
from load_mnist import * 
import hw1_knn  as mlBasics  
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np 
from audioop import cross
import itertools
import time

def load_all_data():
    X_train, y_train = load_mnist('training')
    X_test, y_test = load_mnist('testing')
    
    # Reshape the image data into rows  
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    
    return X_train, y_train, X_test, y_test

def extract_samples_per_class(X_train, y_train, sample_size = 100):
    X_1000 = []
    Y_1000 = []
    
    for i in range(10):
        idxs = np.flatnonzero(y_train == i)
        idxs = np.random.choice(idxs, sample_size, replace=False)
        X_1000.append(X_train[idxs])  
        Y_1000.append(y_train[idxs])    
        
    X_1000 = np.reshape(X_1000, (1000,-1))
    Y_1000 = np.reshape(Y_1000, (1000,-1))
    #Combining training exmaples and corresponding label for shuffling
    combined = np.hstack((X_1000, Y_1000))
    #print X_1000.shape
    np.random.shuffle(combined)
    #Extracting examples and label back
    X_1000 = combined[:,:-1]
    Y_1000 = combined[:, -1]
    
    return X_1000, Y_1000

def cross_validation(X, Y, num_folds=5, k=1):
    # Dividing data into various folds
    X_folds = np.array(np.array_split(X, num_folds))
    y_folds = np.array(np.array_split(Y, num_folds))
    
    # List holding acuracies for k  
    accuracies=[]  
    for i in xrange(num_folds):
        train_id = [x for x in xrange(num_folds) if x != i]
        X_train_data = np.concatenate(X_folds[train_id])
        Y_train_data = np.concatenate(y_folds[train_id])
        dists = mlBasics.compute_euclidean_distances(X_train_data,X_folds[i])
        y_test_pred = mlBasics.predict_labels(dists, Y_train_data, k)
        accuracy = np.mean(y_test_pred==y_folds[i])
        accuracies.append(accuracy)
    
    print 'for k=%d, mean acc=%f '%(k,np.mean(accuracies))
    #for val in accuracies:
    #    print 'accuracy = %f'%(val)
        
    return np.mean(accuracies)

def plot_accuracies(k_accuracies):
    # plot the raw observations
    k_val = np.arange(1,16)
    
    # plot the trend line with error bars that correspond to standard deviation
    plt.plot(k_val,k_accuracies)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()

def test_all_data(X_train, y_train, X_test, y_test, k):
    dists =  mlBasics.compute_euclidean_distances(X_train,X_test) 
    y_test_pred = mlBasics.predict_labels(dists, y_train, k)
    return  np.mean(y_test_pred==y_test)*100   

#source http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

if __name__ == '__main__':
    '''
    (a) Load data - ALL class 
    '''
    X_train, y_train, X_test, y_test = load_all_data()
    
      
    '''
    (b) Load 1000 training example, 100 from each class and visualize 1 and 5 nearest neighbour 
    for first 10 test examples
    '''
    sample_size = 100       #samples per class
    
    X_1000, Y_1000 = extract_samples_per_class(X_train, y_train, sample_size)        
    
    # k=1
    dists =  mlBasics.compute_euclidean_distances(X_1000,X_test) 
    y_test_pred_1 = mlBasics.predict_labels(dists, Y_1000, k=1)  
    print '################## part b #########################'    
    print 'for k=1, {0:0.02f}'.format(np.mean(y_test_pred_1==y_test)*100), "of test examples classified correctly." 
    
    # k=5
    y_test_pred_5 = mlBasics.predict_labels(dists, Y_1000, k=5)      
    print 'for k=5, {0:0.02f}'.format(np.mean(y_test_pred_5==y_test)*100), "of test examples classified correctly."  
    
    
    #Confusion Matrix
    C_1 = metrics.confusion_matrix(y_test, y_test_pred_1)
    C_5 = metrics.confusion_matrix(y_test, y_test_pred_5)
    
    plot_confusion_matrix(C_1, np.arange(10), normalize=False, title = "Confusion Matrix k=1")
    plot_confusion_matrix(C_5, np.arange(10), normalize=False, title = "Confusion Matrix k=5")
 
    '''
    (c) 5 fold cross validation for k = 1 to 15
    '''
    print '################## part c #########################' 
    
    num_folds = 5
    k_val = np.arange(1,16)
    
    k_accuracies = []
    for k in k_val:
         k_accuracies.append(cross_validation(X_1000, Y_1000, num_folds, k))
         
    best_k = np.argmax(k_accuracies) + 1
    acc = np.max(k_accuracies)
   
    
    #plotting   
    plot_accuracies(k_accuracies)
    print 'best_k: ', best_k, ' Accuracy: ', acc
    
    '''
    (d) Using complete data and validating for k=1 and bes_k found in previous part)
    '''
    
    print '################## part d #########################' 
    
    t1 = time.time()
    k1_acc = test_all_data(X_train, y_train, X_test, y_test, 1)
    t2 = time.time()
    best_acc = test_all_data(X_train, y_train, X_test, y_test, best_k)
    t3 = time.time()
    print 'All examples accuracy for k=1 is %f, Time Taken: %f' %(k1_acc, (t2-t1) * 1000.0)
    print 'All examples accuracy for k=%d is %f, Time Taken: %f' %(best_k, best_acc, (t3-t2) * 1000.0)
    print 'All Task finished'
    
    
