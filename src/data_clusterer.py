import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from time import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from data_preprocessor import DataPreprocessor

MOVIES_PATH = "./dataset/movies.csv"
GENRES_PATH = "./dataset/genres.csv"

class DataClusterer:
    def __init__(self, X: pd.DataFrame, max_clusters: int, max_iter: int, init: int):
        self.x = X
        self.max_clusters = max_clusters
        self.max_iters = max_iter
        self.init = init

    def computeKMeans(self, n_clusters: int):
        kmeans = KMeans(n_clusters=n_clusters, max_iter=self.max_iters, n_init=self.init)
        kmeans.fit(self.x)

        y_kmeans = kmeans.predict(self.x) # Let kmeans assign the clusters/labels
        centers = kmeans.cluster_centers_
        inertia = kmeans.inertia_

        # if 1 in y_kmeans:

        #     print(y_kmeans)
        #     silhouette = silhouette_score(self.x, y_kmeans)
            
        # else: silhouette = []

        return centers, y_kmeans, inertia #, silhouette

    def computeKMeansInertia(self, show_plot: bool=True):
        K = [i for i in range(1, self.max_clusters + 1)]
        inertias = []

        for i in range(1, self.max_clusters + 1):
            _, _, inertia, _ = self.computeKMeans(i) # TRAINING PART FOR EACH NUMBER OF CLUSTERS i
            inertias.append(inertia)

        if show_plot:
            plt.plot(K, inertias, 'bx-')
            plt.xlabel('K value')
            plt.ylabel('Inertia')
            plt.show()

        return K, inertias
    
    # def computeKMeansSilhouette(self, show_plot: bool=True):
    #     K = [i for i in range(1, self.max_clusters + 1)]
    #     silhouettes = []

    #     for i in range(1, self.max_clusters + 1):
    #         _, _, _, silhouette = self.computeKMeans(i) # TRAINING PART FOR EACH NUMBER OF CLUSTERS i
    #         silhouettes.append(silhouette)

    #     if show_plot:
    #         plt.plot(K, silhouettes, 'bx-')
    #         plt.xlabel('K value')
    #         plt.ylabel('Silhouette')
    #         plt.show()

    #     return K, silhouettes

    def getClusters(self, n_clusters: int, centers: list, features: list, lsa):
        original_space_centroids = lsa[0].inverse_transform(centers)
        order_centroids = original_space_centroids.argsort()[:, ::-1]

        for i in range(n_clusters):
            print(f"Cluster {i}: ", end="")
            for ind in order_centroids[i, :20]:
                print(f"{features[ind]} ", end="")
            print()