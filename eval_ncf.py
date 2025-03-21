import json

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import manhattan_distances
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity

from ncf import split_data, MovieRatingDataset, train_ncf_model, show_user_history_and_recommend, \
    load_and_preprocess_data, NCF, show_user_history_and_recommend_debug
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


RECOMMENDATION_NUM = 10  # Extract the top_n highest recommendation scores from the model.

sim_methods = ["cosine", "pearson", "manhattan"]
ratings_df, movies_df, user2idx, item2idx, idx2user, idx2item, num_users, num_items, max_timestamp = load_and_preprocess_data(
    './dataset/ratings.csv', './dataset/movies.csv')

# 划分数据
train_df, val_df, test_df = split_data(ratings_df)
train_dataset = MovieRatingDataset(train_df)
val_dataset = MovieRatingDataset(val_df)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# 训练标准NCF模型或加载已有模型
standard_model, standard_optimizer, standard_epochs = train_ncf_model(
    train_loader, val_loader, num_users, num_items,
    num_epochs=30, learning_rate=0.001, load_existing=True
)

# construct mapping {user_id: {movieId: rating, ...}}
user_id_to_ratings = defaultdict(dict)
for i, row in test_df.iterrows():
    user_id_to_ratings[row['userId']][row['movieId']] = row['rating']
# get all the movie id from dataset
movie_list = movies_df['movieId'].unique().tolist()
test_user_ids = test_df['userId'].unique()

like_threshold = 4.0
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
results = defaultdict(list)

# load model
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)
model_file = "model/ncf_model.pth"
model = NCF(len(user2idx), len(item2idx), embedding_dim=32).to(device)
checkpoint = torch.load(model_file, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

for user_id in tqdm(test_user_ids):
    recommendations = show_user_history_and_recommend_debug(user_id, ratings_df, movies_df, user2idx, item2idx, model,
                                                            time_aware=False)

    if recommendations is None:  # TODO
        continue

    genres_recommend = [0] * len(genres)
    genres_truth = [0] * len(genres)
    # Calculate the number of each genre for the recommended movies.

    for _, row in recommendations.iterrows():
        if row['predicted_rating'] < like_threshold:
            continue
        # for all the movies have high recommended score
        for gen in row['genres'].split("|"):
            genres_recommend[genres_to_id[gen]] += 1
        # movie_to_rating_reco[row['movieId']] = row['predicted_rating']
    # 计算用户已有评分的、用户喜欢的电影的各个genres的数量
    for movie, rating in user_id_to_ratings[user_id].items():
        if rating < like_threshold:
            continue
        for genre in movie_to_genres[movie]:
            genres_truth[genres_to_id[genre]] += 1
    # normalize
    genres_recommend, genres_truth = normalize(genres_recommend), normalize(genres_truth)

    for sim_method in sim_methods:
        sim = compute_sim(genres_recommend, genres_truth, movie_list, sim_method=sim_method, skip_building_spare=True)
        results[sim_method].append(sim)

for sim_method, history in results.items():
    print(f"sim_method: {sim_method}, avg: {np.mean(history)}")

with open('./eval_result_3.json', 'w') as f:
    json.dump(results, f, indent=4)
