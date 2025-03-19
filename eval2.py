import itertools
import json
from tqdm import tqdm

from ncf_new import train_time_aware_ncf, recommend_movies, preprocess_data
from user_based_new import *
from typing import Dict, List
from eval import build_sparse_mat, compute_sim
from collections import defaultdict

RECOMMENDATION_NUM = 10 # 取出模型的最大推荐分值top_n

sim_methods = ["cosine", "pearson", "manhattan"]
train_df, val_df, test_df, movies_df, user2idx, item2idx, idx2user, idx2item, num_users, num_items, max_timestamp = preprocess_data(
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
results = []
for sim_method in sim_methods:
    result_entry = {
        "sim_method": sim_method,
    }
    history = []

    for user_id in tqdm(test_user_ids):
        recommendations = recommend_movies(
            model, user_id, train_df, movies_df,
            user2idx, item2idx, top_n=RECOMMENDATION_NUM
        )
        if recommendations is None:  # TODO
            continue
        movie_to_rating_reco = {}
        for _, row in recommendations.iterrows():
            movie_to_rating_reco[row['movieId']] = row['predicted_rating']
        sim = compute_sim(user_id_to_ratings[user_id], movie_to_rating_reco, movie_list, sim_method=sim_method)
        history.append((str(user_id), sim))
    result_entry["history"] = history
    results.append(result_entry)

# 计算Presicion相关metrics
precisions = []
recalls = []
f1_scores = []
n_users_evaluated = 0
k = 10
for user_id in tqdm(test_user_ids):
    recommendations = recommend_movies(
        model, user_id, train_df, movies_df,
        user2idx, item2idx, top_n=k
    )
    if recommendations is None:
        continue
    n_users_evaluated += 1
    movie_to_rating_reco = {}
    for _, row in recommendations.iterrows():
        movie_to_rating_reco[row['movieId']] = row['predicted_rating']
    movies_recommended = set(recommendations['movieId'].tolist())
    movie_to_ratings = user_id_to_ratings[user_id]
    movies_has_ratings = set(movie_to_ratings.keys())

    hit_count = len(movies_recommended.intersection(movies_has_ratings))
    precision = hit_count / k
    recall = hit_count / len(movies_has_ratings)

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    f1_scores.append(f1)

    precisions.append(precision)
    recalls.append(recall)

avg_precision = np.mean(precisions) if precisions else 0
avg_recall = np.mean(recalls) if recalls else 0
avg_f1 = np.mean(f1_scores) if f1_scores else 0

print(f"Precision@{k}: {avg_precision}")
print(f"Recall@{k}: {avg_recall}")
print(f"F1@{k}: {avg_f1}")


with open('./eval_result_2.json', 'w') as f:
    json.dump(results, f, indent=4)

for result in results:
    sim_method, history = result["sim_method"], result["history"]
    history = np.array(history, dtype=np.float64)
    print(f"sim_method: {sim_method}, avg: {np.mean(history[:, 1])}")
