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
from sklearn.model_selection import cross_val_score

# (Question 2)

# Put your funtions here
# ...
def k_fold_cross_validation(cv):
    n_neighbors = np.array([1, 5, 50, 100, 500])
    scores = []
    X, y = make_unbalanced_dataset(3000)

    for n_neighbor in n_neighbors:
        knn = KNeighborsClassifier(n_neighbors = n_neighbor)
        knn_scores = cross_val_score(knn, X, y, cv=cv)

        # We obtain a vector of 10 scores obtained for the 10folds of the data
            # --> average that score : obtained value gives the avg score for
            # n_neighbor
        scores.append(np.mean(knn_scores))
    print("Scores for different values of k : "+str(scores))
    optimal_index = np.argmax(scores)
    optimal = n_neighbors[optimal_index]
    print("The optimal value of k is thus " + str(optimal))


def optimal_value_n_neighbors():
    N = np.array([50, 150, 250, 350, 450, 500])
    n_neighbors = np.array([1, 5, 50, 100, 500])
    test_set = 500
    gen = 10

    for training_set in N:
        mean_test_accuracies = []
        for n_neighbor in n_neighbors:
            accuracies = []
            for i in range(gen):
                X, y = make_unbalanced_dataset(3000)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_set, train_size = training_set)
                fitted_estimator = KNeighborsClassifier(n_neighbors = n_neighbor).fit(X_train, y_train)
                accuracies.append(accuracy_score(y_test, fitted_estimator.predict(X_test)))
            mean_test_accuracies.append(np.mean(accuracies))
        plt.plot(n_neighbors, mean_test_accuracies)
        plt.savefig("accuracy_training_set"+str(training_set)+".pdf")



if __name__ == "__main__":
    # Put your code here

    n_neighbors = np.array([1, 5, 50, 100, 500])

    X, y = make_unbalanced_dataset(3000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.33)

    for n_neighbor in n_neighbors:
        fitted_estimator = KNeighborsClassifier(n_neighbors = n_neighbor).fit(X_train, y_train)
        plot_boundary("n_neighbors"+str(n_neighbor), fitted_estimator, X, y, mesh_step_size = 0.2, title="n_neighbors "+str(n_neighbor))

    k_fold_cross_validation(10)
