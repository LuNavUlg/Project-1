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
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def make_tree(nbSamples = 1500, max_d = None, fname = "", graph = True):
	X,y = make_dataset2(nbSamples)
	X_ls, X_ts, y_ls, y_ts = train_test_split(X, y, train_size =.8)
	estimator = DecisionTreeClassifier(max_depth = max_d).fit(X_ls, y_ls)
	y_pred = estimator.predict(X_ts)
	if graph:
		plot_boundary(fname,estimator, X,y)
	
	return accuracy_score(y_ts, y_pred)


if __name__ == "__main__":

	max_iteration = 25
	depths = [1, 2, 4, 8]
	score = []

	for depth in depths:
		score = make_tree(max_d = depth, fname ="depth_"+str(depth))
		#print("Accuracy for depth "+str(depth) +" : " + str(score))

		score = []
		for i in range(max_iteration):
			score.append(make_tree(max_d = depth, graph = False))
		print("Depth "+str(depth)+" : average accuracy "+ str(np.mean(score)) +", standard deviation : " + str(np.std(score)))
	

	#Depth None
	score = make_tree(fname = "depth_none")
	#print("Accuracy for depth none: " , score)

	score = []
	for i in range(max_iteration):
		score.append(make_tree(graph = False))
	print("Depth none : average accuracy "+ str(np.mean(score)) +", standard deviation : " + str(np.std(score)))

	


