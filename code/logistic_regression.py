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
from sklearn.model_selection import cross_val_score

def conditional_probability_of_positive_class(x, omega0, omega):
    """Computes conditional probability of sample x belonging to the positive
        class knowing parameter theta and data sample X[i, :]

    Parameters
    ----------
    x : vector-like, shape = [n_features]
        The sample.

    omega0 and omega :
        Parameters of the sigmoïd.

    Returns
    -------
    p : conditional probability of sample x belonging to the positive
        class knowing parameter theta and data sample X[i, :]
    """

    p = 1/(1+math.exp(-omega0 - omega.dot(x)))
    return p

def gradient_of_loss_function(X, omega0, omega):
    omega_sum = np.array([0.0, 0.0])
    omega0_sum = 0.0
    N = np.shape(X)[0]
    x_prime = []
    for i in range(N):
        factor = conditional_probability_of_positive_class(X[i], omega0, omega)-y[i]
        omega_sum +=  factor * X[i]
        omega0_sum += factor

    return omega_sum/N, omega0_sum/N

class LogisticRegressionClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_iter=10, learning_rate=1):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.omega0 = None
        self.omega = None

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
        omega0  = 1.0
        omega = np.array((1.0, 1.0))

        for i in range(self.n_iter):
            loss_omega, loss_omega0 = gradient_of_loss_function(X, omega0, omega)
            omega0 = omega0 - self.learning_rate*loss_omega0 # Computes new value w_0
            omega = omega - self.learning_rate*loss_omega # Computes new value w

        # When iterations of gradient descent are done, omega0 and omega are supposed optimal and must be stored
        self.omega0 = omega0
        self.omega = omega

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
        N = np.shape(X)
        proba = self.predict_proba(X)
        for i in range(N[0]):
            if proba[i, 1] >= 0.5:
                y.append(+1)
            else:
                y.append(0)

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

        # COL 0 : NEGATIVE CLASS -- COL 1 : POSITIVE CLASS
        # Warning this has to be coherent with variable proba of predict() method !
        for i in range(N[0]):
            p = conditional_probability_of_positive_class(X[i], omega0, omega)
            row = [1-p, p]
            proba.append(row)

        proba = np.array(proba)
        return proba

if __name__ == "__main__":

    # Put your code here
    X, y = make_unbalanced_dataset(3000)

    logistic_regression = LogisticRegressionClassifier().fit(X[:1000], y[:1000])

    plot_boundary("logistic_regression", logistic_regression, X, y, mesh_step_size=0.1, title="")

    # Here we will report the avg accuracy over five generations of the dataset along with its standard deviation
    gen = 5
    accuracies = []

    for i in range(gen):
        X, y = make_unbalanced_dataset(3000)

        logistic_regression = LogisticRegressionClassifier()
        logistic_regression.fit(X[:1000], y[:1000])
        accuracies.append(accuracy_score(y[1000:], logistic_regression.predict(X[1000:])))

    print("Mean accuracy : ", np.mean(accuracies))
    print("Std : ", np.std(accuracies))
