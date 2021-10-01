"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from plot import plot_boundary
from data import make_dataset1, make_dataset2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score


def make_knn(nbSamples = 1500, n_neigh = 1, fname = "", graph = True, cv = False, data = 2):
	if data == 1:
		X,y = make_dataset1(nbSamples)
	else:
		X,y = make_dataset2(nbSamples)
	if cv:
		return cross_validate(KNeighborsClassifier(n_neighbors = n_neigh), X, y, cv=10)
	else:
		X_ls, X_ts, y_ls, y_ts = train_test_split(X, y, train_size =.8, test_size = .2)
		estimator = KNeighborsClassifier(n_neighbors = n_neigh).fit(X_ls, y_ls)
		y_pred = estimator.predict(X_ts)
		if graph:
			plot_boundary(fname,estimator, X,y)
		return accuracy_score(y_ts, y_pred)


if __name__ == "__main__":

	max_iteration = 25
	n_neighbors = [1, 5, 25, 125, 625, 1200]
	score = []

	for n in n_neighbors:
		score = make_knn(n_neigh = n, fname ="n_neighbors_"+str(n))
		print("Accuracy for n_neighbors "+str(n) +" : " + str(score))

		score = []
		for i in range(max_iteration):
			score.append(make_knn(n_neigh = n, graph = False))
		print("N neighbors "+str(n)+" : average accuracy "+ str(np.mean(score)) +", standard deviation : " + str(np.std(score)))

		scores = make_knn(n_neigh = n, graph = False, cv = True)
		print("10-fold cross validation score : " + str(scores['test_score'].mean()))


	make_knn(n_neigh = 25, fname = "knn_data1", data = 1)
	make_knn(n_neigh = 25, fname = "knn_data2", data = 2)
