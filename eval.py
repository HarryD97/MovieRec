import itertools
import json

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


def compute_sim(movie_to_rating1, movie_to_rating2, movie_list, sim_method):
    mat1 = build_sparse_mat(movie_to_rating1, movie_list)
    mat2 = build_sparse_mat(movie_to_rating2, movie_list)
    if sim_method == 'cosine':
        sim = cosine_similarity(mat1, mat2)[0][0]
    elif sim_method == 'pearson':
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


model_types = ["cf", "svd"]
sim_methods = ["cosine", "pearson", "manhattan"]
cf = EnhancedCF(n_sim_user=30, n_rec_movie=10, pivot=0.8, n_factors=35, sim_method='cosine')
cf.get_dataset()
# 获取所有的movie id
movie_set = set()
for movie_to_rating in cf.testSet.values():
    movie_set.update(movie_to_rating.keys())
for movie_to_rating in cf.trainSet.values():
    movie_set.update(movie_to_rating.keys())
movie_list = list(movie_set)

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

        mat1 = build_sparse_mat(moive_to_rating, movie_list)
        mat2 = build_sparse_mat(recommendations, movie_list)
        if sim_method == 'cosine':
            sim = cosine_similarity(mat1, mat2)[0][0]
        elif sim_method == 'pearson':
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