import pandas as pd
import numpy as np
import sys
import datetime
import math
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import time
from scipy import stats
import statsmodels.api as sm
from mlxtend.feature_selection import SequentialFeatureSelector
from scipy.stats import pearsonr # get pearsonsr
from pandas.api.types import is_numeric_dtype

# ---- sklearn ----
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors # importing the library
from sklearn.cluster import DBSCAN
from sklearn import datasets
from sklearn.cluster import KMeans


def kscan(data, graph=True, kmeans_n_init=100, kmeans_max_iter=1000, kmeans_tol=0.0001):
    """Compute data proximity from each other using Nearest Neighbors"""
    neighb = NearestNeighbors(algorithm="brute")  # creating an object of the NearestNeighbors class
    nbrs = neighb.fit(data)  # fitting the data to the object
    distances, indices = nbrs.kneighbors(data)  # finding the nearest neighbours

    """Automate Approximation of Eps"""
    # Remove 0 indices and distance values
    unlisted_distance = [item for sublist in distances for item in sublist if item != 0]
    sort_distances = sorted(unlisted_distance)

    dif1 = list()
    for i in range(1, len(sort_distances)):
        dif1.append(sort_distances[i] - sort_distances[i - 1])

    sort_dif1 = sorted(dif1)
    dif2 = list()
    for i in range(1, len(sort_distances) - 1):
        dif2.append(sort_dif1[i] - sort_dif1[i - 1])

    best_pt = next((i for i in dif2 if i >= 0.01), None)

    # Epsilon
    automated_eps = (sort_distances[dif2.index(best_pt) + 2] + sort_distances[dif2.index(best_pt) + 1]) / 2

    if graph == True:
        # Sort and plot the distances results
        distances = np.sort(distances, axis=0)  # sorting the distances
        distances = distances[:, 1]  # taking the second column of the sorted distances
        plt.rcParams['figure.figsize'] = (5, 3)  # setting the figure size
        plt.plot(distances)  # plotting the distances
        plt.show()  # showing the plot
    else:
        pass

    # Automate approximation of MinPts
    auto_dimension = data.shape[1]

    if auto_dimension <= 2:
        minpts = 4
    else:
        minpts = auto_dimension * 2

    # Cluster the data
    dbscan = DBSCAN(eps=automated_eps, min_samples=minpts).fit(data)  # fitting the model
    labels = dbscan.labels_  # getting the labels
    # Determine number of unique clusters
    unique_clusters = len(set(list(labels)))

    if graph == True:
        # Plot the clusters
        plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap="plasma")  # plotting the clusters
        plt.xlabel("Income")  # X-axis label
        plt.ylabel("Spending Score")  # Y-axis label
        plt.show()  # showing the plot
    else:
        pass

    # run K-Means
    kmeanModel_best = KMeans(n_clusters=unique_clusters, n_init=kmeans_n_init, max_iter=kmeans_max_iter, tol=kmeans_tol).fit(data)

    return [automated_eps, unique_clusters, kmeanModel_best]

