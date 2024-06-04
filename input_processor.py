import sys
import pandas as pd
import numpy as np

from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

PATH_TO_SRC = "./src"

if PATH_TO_SRC not in sys.path:
  sys.path.append(PATH_TO_SRC)

from config import MOVIES_PATH, GENRES_PATH, TEST_SIZE, RANDOM_STATE, MAX_DF, MIN_DF, STOP_WORDS, N_COMPONENTS, NORMALIZER_COPY, N_CLUSTERS, MAX_ITER, N_INIT, OPTIMAL_K
from data_preprocessor import DataPreprocessor
from data_clusterer import DataClusterer

'''
def vectorizeInput(input: str, stop_words: str=STOP_WORDS):
    d = {'description': [input]}
    X = pd.DataFrame(data=d)
    X_desc_only = X.description

    vectorizer = TfidfVectorizer(
            stop_words=stop_words,
        )

    X_tfidf = vectorizer.fit_transform(X_desc_only)
    print(X_tfidf.shape)

    return X_tfidf

def reduceDim(X_tfidf: pd.DataFrame, num_comp: int=N_COMPONENTS, normalizer_copy: bool=NORMALIZER_COPY) -> np.array:
    lsa = make_pipeline(TruncatedSVD(n_components=num_comp), Normalizer(copy=normalizer_copy))

    X_lsa = lsa.fit_transform(X_tfidf)
    # explained_variance = lsa[0].explained_variance_ratio_.sum()

    return X_lsa

def findNearestCentroid(X_lsa: pd.DataFrame) -> int:
    assert X_lsa.shape[0] == 1

    centroids = np.load('cluster_centers.npy')
    cluster_min = 0
    min_distance = 0xFFFFFFFF

    for i in range(len(centroids)):
        dist = np.linalg.norm(X_lsa[0] - centroids[i])
        # print("Cluster: ", i)
        # print("Distance: ", dist)

        if dist < min_distance:
            min_distance = dist
            cluster_min = i

    return cluster_min

def computeDistances(X_lsa: pd.DataFrame, cluster: int) -> pd.DataFrame:
    assert X_lsa.shape[0] == 1

    vectorized = np.load('vectorized_data.npy')
    clustered_movies = pd.read_csv('clustered_movies.csv')
    distances = []

    for i in range(len(vectorized)):
        dist = np.linalg.norm(X_lsa[0] - vectorized[i])
        distances.append(round(dist, 4))

    clustered_movies = clustered_movies.assign(similarity=distances)
    recommendations = clustered_movies.loc[clustered_movies['cluster'] == cluster].sort_values('similarity', ascending=True).head(20)

    return recommendations # recommendations.to_csv('recommendations.csv')
'''

def processInput(input: str) -> int:
    '''
        X_tfidf = vectorizeInput(input)

        X_lsa = reduceDim(X_tfidf)
        cluster = findNearestCentroid(X_lsa)

        return X_lsa, cluster
    '''
    preprocessor = DataPreprocessor(MOVIES_PATH, GENRES_PATH)
    preprocessor.handleDataFrame()

    preprocessor.df_movies.loc[-1] = [None, None, None, input, None] # Set input as first row of data
    preprocessor.df_movies.index += 1
    preprocessor.df_movies.sort_index()

    X_desc_only = preprocessor.df_movies.description

    X_tfidf, features = preprocessor.vectorizeData(X_desc_only, MAX_DF, MIN_DF) # Vectorize
    X_lsa, lsa = preprocessor.reduceDim(X_tfidf, N_COMPONENTS, NORMALIZER_COPY) # DimRed

    clusterer = DataClusterer(X_lsa, N_CLUSTERS, MAX_ITER, N_INIT) 
    centers, y_kmeans, inertia = clusterer.computeKMeans(OPTIMAL_K) # Cluster
    clusterer.getClusters(OPTIMAL_K, centers, features, lsa)

    distances = []

    for i in range(len(X_lsa)):
        dist = np.linalg.norm(X_lsa[0] - X_lsa[i])
        distances.append(round(dist, 4))

    X_clustered  = preprocessor.df_movies.assign(similarity=distances)
    X_clustered = X_clustered.assign(cluster=y_kmeans)
    recommendations = X_clustered.loc[X_clustered['cluster'] == X_clustered.loc[0]['cluster']].sort_values('similarity', ascending=True).head(20)

    return recommendations

    






    

