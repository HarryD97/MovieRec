import json

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import manhattan_distances
from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from ncf_new import train_time_aware_ncf, recommend_movies, preprocess_data
from collections import defaultdict
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

RECOMMENDATION_NUM = 10  # 取出模型的最大推荐分值top_n

sim_methods = ["cosine", "pearson", "manhattan"]
train_df, val_df, test_df, movies_df, user2idx, item2idx, idx2user, idx2item, num_users, num_items, timestamp_info = preprocess_data(
    './dataset/ratings.csv', './dataset/movies.csv')

model, optimizer, trained_epochs = train_time_aware_ncf(
    train_df, val_df, num_users, num_items,
    batch_size=256, num_epochs=20, learning_rate=0.001,
    load_existing=True
)

# 构造{user_id: {movieId: rating, ...}}
user_id_to_ratings = defaultdict(dict)
for i, row in test_df.iterrows():
    user_id_to_ratings[row['userId']][row['movieId']] = row['rating']
# 获取所有的movie_id
movie_list = movies_df['movieId'].unique().tolist()
test_user_ids = test_df['userId'].unique()

# 计算相似度相关的metrics
# results = []
# for sim_method in sim_methods:
#     result_entry = {
#         "sim_method": sim_method,
#     }
#     history = []
#
#     for user_id in tqdm(test_user_ids):
#         recommendations = recommend_movies(
#             model, user_id, train_df, movies_df,
#             user2idx, item2idx, top_n=RECOMMENDATION_NUM
#         )
#         if recommendations is None:  # TODO
#             continue
#         movie_to_rating_reco = {}
#         for _, row in recommendations.iterrows():
#             movie_to_rating_reco[row['movieId']] = row['predicted_rating']
#         sim = compute_sim(user_id_to_ratings[user_id], movie_to_rating_reco, movie_list, sim_method=sim_method)
#         history.append((str(user_id), sim))
#     result_entry["history"] = history
#     results.append(result_entry)

like_threshold = 4.0
# 获取数据集中所有的genre
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
results = defaultdict(list)
for user_id in tqdm(test_user_ids):
    recommendations = recommend_movies(
        model, user_id, train_df, movies_df,
        user2idx, item2idx, timestamp_info, top_n=RECOMMENDATION_NUM, use_current_time=True
    )
    if recommendations is None:  # TODO
        continue

    genres_recommend = [0] * len(genres)
    genres_truth = [0] * len(genres)
    # 计算推荐的电影的各个genres的数量

    for _, row in recommendations.iterrows():
        if row['predicted_rating'] < like_threshold:
            continue
        # 喜欢这个电影
        for gen in row['genres'].split("|"):
            genres_recommend[genres_to_id[gen]] += 1
        # movie_to_rating_reco[row['movieId']] = row['predicted_rating']
    # 计算用户已有评分的、用户喜欢的电影的各个genres的数量
    for movie, rating in user_id_to_ratings[user_id].items():
        if rating < like_threshold:
            continue
        for genre in movie_to_genres[movie]:
            genres_truth[genres_to_id[genre]] += 1
    # 归一化
    genres_recommend, genres_truth = normalize(genres_recommend), normalize(genres_truth)

    for sim_method in sim_methods:
        sim = compute_sim(genres_recommend, genres_truth, movie_list, sim_method=sim_method, skip_building_spare=True)
        results[sim_method].append(sim)

for sim_method, history in results.items():
    print(f"sim_method: {sim_method}, avg: {np.mean(history)}")

# 计算Presicion相关metrics
# precisions = []
# recalls = []
# f1_scores = []
# n_users_evaluated = 0
k = 10
# for user_id in tqdm(test_user_ids):
#     recommendations = recommend_movies(
#         model, user_id, train_df, movies_df,
#         user2idx, item2idx, top_n=k
#     )
#     if recommendations is None:
#         continue
#     n_users_evaluated += 1
#     movie_to_rating_reco = {}
#     for _, row in recommendations.iterrows():
#         movie_to_rating_reco[row['movieId']] = row['predicted_rating']
#     movies_recommended = set(recommendations['movieId'].tolist())
#     movie_to_ratings = user_id_to_ratings[user_id]
#     movies_has_ratings = set(movie_to_ratings.keys())
#
#     hit_count = len(movies_recommended.intersection(movies_has_ratings))
#     precision = hit_count / k
#     recall = hit_count / len(movies_has_ratings)
#
#     if precision + recall > 0:
#         f1 = 2 * precision * recall / (precision + recall)
#     else:
#         f1 = 0
#     f1_scores.append(f1)
#
#     precisions.append(precision)
#     recalls.append(recall)

# avg_precision = np.mean(precisions) if precisions else 0
# avg_recall = np.mean(recalls) if recalls else 0
# avg_f1 = np.mean(f1_scores) if f1_scores else 0

# print(f"Precision@{k}: {avg_precision}")
# print(f"Recall@{k}: {avg_recall}")
# print(f"F1@{k}: {avg_f1}")

with open('./eval_result_2.json', 'w') as f:
    json.dump(results, f, indent=4)

# for result in results:
#     sim_method, history = result["sim_method"], result["history"]
#     history = np.array(history, dtype=np.float64)
#     print(f"sim_method: {sim_method}, avg: {np.mean(history[:, 1])}")
