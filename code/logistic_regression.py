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

def conditional_propabilty_of_positive_class(x, theta):
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

    w_0 = theta[0] # real value
    w = theta[1] # vector

    p = 1/(1+exp(-w_0 - np.dot(np.transpose(w), x)))
    return p

def gradient_of_loss_function(X, theta):
    sum = 0
    N = np.shape(X)
    for i in range(N):
        x_prime = np.transpose(np.append(1, X[i]))
        sum = sum + (conditional_propabilty_of_positive_class(X[i], )-y[i])*x_prime[i]
    return (1/N)*sum

def loss_function(theta, X):
    sum = 0
    N = np.shape(X)
    for i in range(N):
        sum = sum + log((conditional_propabilty_of_positive_class(X[i], theta)))
    return -sum/N

class LogisticRegressionClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_iter=10, learning_rate=1):
        self.n_iter = n_iter
        self.learning_rate = learning_rate


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
        X = np.asarray(X, dtype=np.float)
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
        theta = []
        for i in range(self.n_iter):
            theta_new = theta_old - self.learning_rate*gradient_loss(theta_old)
            np.append(theta, theta_new)
            theta_old = theta_new

        # Now compute loss function for all values of theta
        loss_functions = []
        for theta_val in theta:
            np.append(loss_functions, loss_function(theta_val, X))

        # Find minimum loss function
        min_index = argmin(loss_functions)
        # Find corresponding theta value
        optimal_theta = theta[min_index]

        self.BaseEstimator.set_params(optimal_theta)
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
        theta = self.BaseEstimator.get_params()
        for i in range(size[0]):
            if conditional_propabilty_of_positive_class(X[i], theta) >= 0.5:
                y[i] = +1
            else:
                y[i] = -1

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
        pass

if __name__ == "__main__":
    pass
