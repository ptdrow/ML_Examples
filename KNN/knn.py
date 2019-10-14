#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 19:07:16 2019

@author: ptdrow
"""
import math
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

#Calculate Euclidean distance:
def euclidean_dist(xi,xj):
    """
    Calculates the euclidean distance in a N-dimensional space between the points xi and xj
    
    Parameters:
        xi, xj: numeric list or tuple representing each point.
        
    Returns:
        distance (float): the euclidean distance between the points.
        
    Usage:
        
        >>> x1 = (2,1,1)
        >>> x2 = (1,0,0)
        >>> euclidean_dist(x1,x2)
        1.7320508075688772
        
    """
    
    distance_squared = 0
    for d in range(len(xi)):
        distance_squared += (xj[d] - xi[d]) ** 2
    return math.sqrt(distance_squared)
 

def calc_all_distances(xi, data, columns):
    """
    Calculates the Euclidean distance between a point and all the points in the data.
    
    Parameters:
        xi      : Numeric list or tuple representing the point.
        data    : Pandas dataframe with the training data.
                  Each row will represent a point in the N-dimensional space.
        columns : A list of the columns name to take as the dimensions of the Euclidean space. 
        
    Returns:
        distances (Series): the euclidean distances between the points.
        
    """
    #Convert data frame to points in n-dimensional space
    points = data[columns].apply(tuple,axis=1)
    #Calculate distance for each row
    distances = points.apply(lambda x: euclidean_dist(xi,x))
    return distances


def predict_knn(xi, data,columns, target_column, k):
    """
    Predicts the label for a given point by taking votes from its k-nearest neigbors
    
    Parameters:
        xi           : Numeric list or tuple representing the point.
        data         : Pandas dataframe with the training data.
                       Each row will represent a point in the N-dimensional space.
        columns      : A list of the columns names to take as the dimensions of the Euclidean space. 
        target_column: The name of the target feature
        k            : The number of nearest neighbors to take into account
        
    Returns:
        prediction (str): the predicted label for the point
    
    """
    
    all_distances = calc_all_distances(xi,data,columns)
    k_nearest = list(all_distances.sort_values()[:k].index)
    k_nearest_labels = data.loc[k_nearest, target_column]
    poll = Counter(k_nearest_labels)
    return poll.most_common()[0][0]

def predict_all(data, columns, target_column, k):
    """
    Predicts a label for each point in the data set, by looking at its k-nearest neighbors
    
    Parameters:
        data         : Pandas dataframe with the training data.
                       Each row will represent a point in the N-dimensional space.
        columns      : A list of the columns names to take as the dimensions of the Euclidean space. 
        target_column: The name of the target feature
        k            : The number of nearest neighbors to take into account
        
    Returns:
        predictions (list): the predicted labels for all the points
    """
    
    predictions = list()
    for row in data.itertuples():
        dict_row = row._asdict()
        xi = list()
        for col in columns:
            xi.append(dict_row[col])
        predictions.append(predict_knn(xi, 
                                       data.drop(index=row.Index), 
                                       columns, 
                                       target_column, 
                                       k))
    return predictions
    
def accuracy(true_label,predicted_label):
    """
    Calculates the accuracy for a model
    
    Parameters:
        true_label      : a list or series with the true labels of the points
        predicted_label : a list or series with the predicted label of the points
    
    Returns:
        accuracy: the accuracy score for the model
        
    """
    return sum(true_label == predicted_label)/len(true_label)
    
    
iris_data = pd.read_csv('IRIS.csv')
feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
target_column = 'species'

accuracies = list()
ks = [k for k in range(1,22)]
for k in ks:
    accuracies.append(accuracy(iris_data.loc[:,target_column],
                                  predict_all(iris_data,feature_columns, target_column,k)))
    print('\r{0} Done'.format(k), end='')        
    

plt.clf()
plt.plot(ks,accuracies, 'go')  # green dots
plt.show()