"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math

from plot import plot_boundary
from data import make_balanced_dataset, make_unbalanced_dataset
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split, cross_val_score

def conditional_propability_of_positive_class(x, omega0, omega):
    """Computes conditional probability of sample x belonging to the positive
        class knowing parameter theta and data sample X[i, :]
    Parameters
    ----------
    x : vector-like, shape = [n_features]
        The sample.
    theta : vector-like, [omega_0, omega^T (vetor)]
        Parameters of the sigmoïd.
    Returns
    -------
    p : conditional probability of sample x belonging to the positive
        class knowing parameter theta and data sample X[i, :]
    """

    p = 1/(1+math.exp(-omega0 - np.dot(np.transpose(omega), x)))
    return p

def gradient_of_loss_function(X, omega0, omega):
    sum = 0
    N = np.shape(X)
    x_prime = []
    for i in range(N[0]):
        x_prime = np.append(x_prime, np.transpose(np.append(X[i], 1)))
        sum = sum + np.dot((conditional_propability_of_positive_class(X[i], omega0, omega)-y[i]), x_prime[i])

    return sum/N[0]

def loss_function(X, omega0, omega):
    sum = 0
    N = np.shape(X)
    for i in range(N[0]):
        sum = sum + np.log((conditional_propability_of_positive_class(X[i], omega0, omega)))
    return -sum/N

class LogisticRegressionClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_iter=10, learning_rate=1):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.omega0 = 0
        self.omega = []

    def fit(self, X, y):
        """Fit a logistic regression models on (X, y)
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.
        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")
        n_instances, n_features = X.shape

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        n_classes = len(np.unique(y))
        if n_classes != 2:
            raise ValueError("This class is only dealing with binary "
                             "classification problems")


        # TODO insert your code here

        # Gradient descent to compute possible values of theta
        omega0s = [] # vector of real values of omega0
        omegas = [] # list of vectors
        omega0_old  = 1
        omega_old = (1, 1)

        for i in range(self.n_iter):
            omega0_new = omega0_old - self.learning_rate*gradient_of_loss_function(X, omega0_old, omega_old) # Computes new value w_0
            omega_new = omega_old - self.learning_rate*gradient_of_loss_function(X, omega0_old, omega_old) # Computes new values w

            omega0s.append(omega0_new)
            omegas.append(omega_new)

            omega0_old  = omega0_new
            omega_old = omega_new

        # Now compute loss function for all values of theta
        loss_functions = []
        for i in range(self.n_iter):
            omega0 = omega0s[i]
            omega = omegas[i]
            value = loss_function(X, omega0, omega)
            loss_functions.append(value)

        # Find minimum loss function
        min_index = np.argmin(loss_functions)
        # Find corresponding theta value
        optimal_omega0 = omega0s[min_index]
        optimal_omega = omegas[min_index]

        self.omega0 = optimal_omega0
        self.omega = optimal_omega

        return self


    def predict(self, X):
        """Predict class for X.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """

        # TODO insert your code here
        y = []
        size = np.shape(X)
        proba = self.predict_proba(X)
        for i in range(size[0]):
            if proba[i][1] >= 0.5:
                y.append(1)
            else:
                y.append(-1)

        return y

    def predict_proba(self, X):
        """Return probability estimates for the test data X.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        # TODO insert your code here

        proba = []
        N = np.shape(X)
        omega0 = self.omega0
        omega = self.omega

        for i in range(N[0]):
            row = [conditional_propability_of_positive_class(X[i], omega0, omega), 1-conditional_propability_of_positive_class(X[i], omega0, omega)]
            proba.append(row)

        proba = np.array(proba)
        return proba

if __name__ == "__main__":
    accuracies = []
    plot = True
    iteration = 100
    # Put your code here
    for i in range(iteration):
        X, y = make_unbalanced_dataset(3000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.33)
        logistic_regression = LogisticRegressionClassifier()
        logistic_regression.fit(X_train, y_train)
        if plot is True:
            plot_boundary("logistic_regression", logistic_regression, X, y, mesh_step_size=0.1, title="")
            plot = False
        accuracies.append(accuracy_score(y_test, logistic_regression.predict(X_test)))
    print("Accuracy:",np.mean(accuracies))
    print("Std:",np.std(accuracies))
