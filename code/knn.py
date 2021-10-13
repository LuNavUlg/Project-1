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
from sklearn.model_selection import cross_val_score

# (Question 2)

# Put your funtions here
# ...
def k_fold_cross_validation(cv):
    n_neighbors = np.array([1, 5, 50, 100, 500])
    scores = []
    knn_scores = []
    X, y = make_unbalanced_dataset(3000)

    for n_neighbor in n_neighbors:
        for cut in range(cv):
            X_test, y_test = [], []
            X_training, y_training = [], []

            cut_1, cut_2 = cut*300, (cut+1)*300

            X_test, y_test = X[cut_1:cut_2], y[cut_1:cut_2]
            X_train_part1, X_train_part2 = X[:cut_1], X[cut_2:]
            y_train_part1, y_train_part2 = y[:cut_1], y[cut_2:]

            X_training = np.concatenate((X_train_part1, X_train_part2))
            y_training = np.concatenate((y_train_part1, y_train_part2))

            fitted_estimator = KNeighborsClassifier(n_neighbors = n_neighbor).fit(X_training, y_training)
            knn_scores.append(accuracy_score(y_test, fitted_estimator.predict(X_test)))

        scores.append(np.mean(knn_scores))
    print("Scores for different values of k : "+str(scores))
    optimal_index = np.argmax(scores)
    optimal = n_neighbors[optimal_index]
    print("The optimal value of k is thus " + str(optimal))


def optimal_value_n_neighbors():
    N = np.array([50, 150, 250, 350, 450, 500])
    test_set = 500
    gen = 10

    for train_set in N:
        mean_test_accuracies = []
        plot_neighbors = []
        for n_neighbor in range(1, test_set + 1, 1):
            if train_set >= n_neighbor:
                accuracies = []
                for i in range(gen):
                    X, y = make_unbalanced_dataset(test_set + train_set)
                    X_training, y_training = X[:training_set], y[:train_set]
                    X_test, y_test = X[train_set:], y[train_set:]
                    fitted_estimator = KNeighborsClassifier(n_neighbors = n_neighbor).fit(X_training, y_training)
                    accuracies.append(accuracy_score(y_test, fitted_estimator.predict(X_test)))
                mean_test_accuracies.append(np.mean(accuracies))
                plot_neighbors.append(n_neighbor)
        plt.plot(plot_neighbors, mean_test_accuracies, label="Training set "+ str(train_set))
    plt.xlabel("Number of neighbors")
    plt.ylabel("Accuracies")
    plt.legend(loc='upper right', shadow=True)
    plt.savefig("accuracy_training_set"+".pdf")


if __name__ == "__main__":
    # Put your code here
    X, y = make_unbalanced_dataset(3000)
    n_neighbors = np.array([1, 5, 50, 100, 500])

    for n_neighbor in n_neighbors:
        fitted_estimator = KNeighborsClassifier(n_neighbors = n_neighbor).fit(X[:1000], y[:1000])
        plot_boundary("n_neighbors"+str(n_neighbor), fitted_estimator, X, y, mesh_step_size = 0.2, title="n_neighbors "+str(n_neighbor))
    #optimal_value_n_neighbors()
    k_fold_cross_validation(10)

