"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from plot import plot_boundary
from data import make_balanced_dataset, make_unbalanced_dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split 


# (Question 1)

if __name__ == "__main__":

    split_values = [2, 8, 32, 64, 128, 500]

    X, y = make_unbalanced_dataset(3000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.33)
    
    for split_value in split_values:
        fitted_estimator = DecisionTreeClassifier(min_samples_split = split_value).fit(X_train, y_train)
        plot_boundary("split_value"+str(split_value), fitted_estimator, X, y, mesh_step_size = 0.2, title="")