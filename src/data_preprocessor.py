import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from time import time

class DataPreprocessor:
    def __init__(self, path_to_movies: str, path_to_genres: str):
        self.movies_path = path_to_movies
        self.genres_path = path_to_genres

        self.df_movies = pd.read_csv(self.movies_path)
        self.df_genres = pd.read_csv(self.genres_path)

    def countGenres(self):
        df_unique_genres = self.df_genres.drop_duplicates("id")
        df_count = df_unique_genres.groupby('genre').nunique()

        print("# of Genres: ", df_count.shape[0])
        print("Unique Genres: ", df_count[0])

        return df_count

    def handleDataFrame(self):
        df_unique_genres = self.df_genres.drop_duplicates("id")
        self.df_movies = pd.merge(df_unique_genres, self.df_movies, on="id")

        self.df_movies.drop(columns=["date", "tagline", "minute", "minute"], inplace=True)
        self.df_movies = self.df_movies[self.df_movies['description'].notna()]

        print("Length: ", self.df_movies.shape[0])
        print("Columns: ", self.df_movies.columns)
    
    def splitData(self, test_size: float, rand_state: int):
        assert hasattr(self.df_movies, 'description') and hasattr(self.df_movies, 'genre')

        X = self.df_movies.description
        y = self.df_movies.genre

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_state)

        return X_train, X_test, y_train, y_test
    
    def vectorizeData(self, X: pd.DataFrame, max_df: float, min_df: int, stop_words: str):
        vectorizer = TfidfVectorizer(
            max_df=max_df,
            min_df=min_df,
            stop_words=stop_words,
        )

        t0 = time()
        X_tfidf = vectorizer.fit_transform(X)
        t1 = time() - t0

        print(f"Vectorized in {t1:.3f} s")
        print(f"# of Samples: {X_tfidf.shape[0]}")
        print(f"# of Features: {X_tfidf.shape[1]}")
        print(f"% of Nonzero Entries: {X_tfidf.nnz / np.prod(X_tfidf.shape):.3f}")

        return X_tfidf