MOVIES_PATH = "./dataset/movies.csv"
GENRES_PATH = "./dataset/genres.csv"

TEST_SIZE = 0.3
RANDOM_STATE = 42

MAX_DF = 0.3 # Ignore terms that appear in > 30% of the entries
MIN_DF = 3 # Ignore terms that appear in < 3 entries
STOP_WORDS = "english"

N_COMPONENTS = 50
NORMALIZER_COPY = False

N_CLUSTERS = 20
MAX_ITER = 100
N_INIT = 1

OPTIMAL_K = 15

CUSTOM_STOP_WORDS = {'lives', 'life', 'director', 'directed', 'film', 'films', 'filmmaker'}
