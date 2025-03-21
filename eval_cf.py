import itertools
import json
from collections import defaultdict

from tqdm import tqdm

from user_based_new import *
from typing import Dict, List


def build_sparse_mat(movie_rating_data: Dict[str, float], movie_list: List[str]):
    movie_to_idx = {m: j for j, m in enumerate(movie_list)}
    data, rows, cols = [], [], []
    if isinstance(movie_rating_data, dict):
        for movie, rating in movie_rating_data.items():
            cols.append(movie_to_idx[movie])
            data.append(rating)
    elif isinstance(movie_rating_data, list):
        for movie, score in movie_rating_data:
            cols.append(movie_to_idx[movie])
            data.append(score)
    else:
        print("error: unknown movie_data data type")
        exit(1)
    rows = [0] * len(cols)
    sparse_mat = csr_matrix((data, (rows, cols)), shape=(1, len(movie_list)))
    return sparse_mat

def compute_sim(movie_to_rating1, movie_to_rating2, movie_list, sim_method, skip_building_spare=False):
    if skip_building_spare:
        mat1, mat2 = movie_to_rating1, movie_to_rating2
    else:
        mat1 = build_sparse_mat(movie_to_rating1, movie_list)
        mat2 = build_sparse_mat(movie_to_rating2, movie_list)
    if sim_method == 'cosine':
        sim = cosine_similarity(mat1, mat2)[0][0]
    elif sim_method == 'pearson':
        if skip_building_spare:
            dense_mat1, dense_mat2 = movie_to_rating1, movie_to_rating2
        else:
            dense_mat1, dense_mat2 = mat1.toarray(), mat2.toarray()
        mask = np.logical_and(dense_mat1 > 0, dense_mat2 > 0)
        if np.sum(mask) < 2:  # ref: EnhancedCF._compute_similarity
            sim = 0
        else:
            u1_common = dense_mat1[mask]
            u2_common = dense_mat2[mask]
            try:
                corr, _ = pearsonr(u1_common, u2_common)
                if np.isnan(corr):
                    corr = 0.0
            except:
                corr = 0.0
            sim = (corr + 1) / 2.0
    elif sim_method == 'manhattan':
        sim = manhattan_distances(mat1, mat2)[0][0]
    return sim

def normalize(arr):
    arr = np.array(arr)
    s = arr.sum()
    if s != 0:
        arr = arr / s
    return arr.reshape(1, -1)


model_types = ["cf", "svd"]
sim_methods = ["cosine", "pearson", "manhattan"]
cf = EnhancedCF(n_sim_user=30, n_rec_movie=10, pivot=0.8, n_factors=35, sim_method='cosine')
cf.get_dataset()
# get all the movie id from dataset
movie_set = set()
for movie_to_rating in cf.testSet.values():
    movie_set.update(movie_to_rating.keys())
for movie_to_rating in cf.trainSet.values():
    movie_set.update(movie_to_rating.keys())
movie_list = list(movie_set)

like_threshold = 4.0
movies_df = pd.read_csv('./dataset/movies.csv')
# get all the genres from dataset
genres = movies_df['genres'].tolist()
genres_set = set()
for g in genres:
    genres_set.update(list(g.split("|")))
genres = list(genres_set)
print("all genres:", genres)
genres_to_id = {genre: i for i, genre in enumerate(genres)}
# map movieId -> genre list
movie_to_genres = defaultdict(list)
for _, row in movies_df.iterrows():
    movie_to_genres[row['movieId']] = row['genres'].split('|')

results = []

for model_type, sim_method in itertools.product(model_types, sim_methods):
    cf.load_model(model_type=model_type, sim_method=sim_method)
    result_entry = {
        "model_type": model_type,
        "sim_method": sim_method,
    }
    history = []
    for user, moive_to_rating in tqdm(cf.testSet.items()):
        recommendations = cf.recommend(user)
        genres_recommend = [0] * len(genres)
        genres_truth = [0] * len(genres)
        for movie, rating in recommendations:
            movie = int(float(movie))
            if rating < like_threshold:
                continue
            for g in movie_to_genres[movie]:
                genres_recommend[genres_to_id[g]] += 1
        for movie, rating in movie_to_rating.items():
            movie = int(float(movie))
            if rating < like_threshold:
                continue
            for g in movie_to_genres[movie]:
                genres_truth[genres_to_id[g]] += 1
        genres_recommend, genres_truth = normalize(genres_recommend), normalize(genres_truth)
        sim = compute_sim(genres_recommend, genres_truth, movie_list, sim_method, skip_building_spare=True)
        history.append((user, sim))
        # print(f"user={user}, modle_type={model_type}, sim_method={sim_method}, sim={sim}, type={type(sim)}")
    result_entry["history"] = history
    results.append(result_entry)

with open('./eval_result.json', 'w') as f:
    json.dump(results, f, indent=4)

for result in results:
    model_type, sim_method, history = result["model_type"], result["sim_method"], result["history"]
    history = np.array(history, dtype=np.float64)
    print(f"model_type: {model_type}, sim_method: {sim_method}, avg: {np.mean(history[:, 1])}")