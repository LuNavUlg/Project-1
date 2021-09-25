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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


# (Question 2)

# Put your funtions here
# ...


if __name__ == "__main__":
    # Put your code here

    n_neighbors = [1, 5, 50, 100, 500]

    X, y = make_unbalanced_dataset(3000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.33)

    for n_neighbor in n_neighbors:
        fitted_estimator = KNeighborsClassifier(n_neighbors = n_neighbor).fit(X_train, y_train)
        plot_boundary("n_neighbors"+str(n_neighbor), fitted_estimator, X, y, mesh_step_size = 0.2, title="n_neighbors "+str(n_neighbor))
