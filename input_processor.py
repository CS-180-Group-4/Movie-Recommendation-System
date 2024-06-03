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

from config import MAX_DF, MIN_DF, STOP_WORDS, N_COMPONENTS, NORMALIZER_COPY

def vectorizeInput(input: str, stop_words: str=STOP_WORDS):
    d = {'description': [input]}
    X = pd.DataFrame(data=d)
    X_desc_only = X.description

    vectorizer = TfidfVectorizer(
            stop_words=stop_words,
        )

    X_tfidf = vectorizer.fit_transform(X_desc_only)

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
        print("Cluster: ", i)
        print("Distance: ", dist)

        if dist < min_distance:
            min_distance = dist
            cluster_min = i

    return cluster_min

def processInput(input: str) -> int:
    X_tfidf = vectorizeInput(input)

    X_lsa = reduceDim(X_tfidf)
    cluster = findNearestCentroid(X_lsa)
        
    return cluster




    

