#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances
from scipy.stats import pearsonr
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import psycopg2
from psycopg2 import Error
from db_connect import get_db_connection
from db_query import query_movie_details, query_user_ratings

MODEL_BASE = "model/cf_model"
SVD_MODEL_BASE = "model/svd_model"

def load_data_from_db():
    conn = get_db_connection()
    print(conn)
    if conn is None:
        print("Failed to connect to the database.")
        return None, None

    try:
        ratings_query = 'SELECT "userId", "movieId", "rating", "timestamp" FROM ratings'
        movies_query = 'SELECT "movieId", "title", "genres" FROM movies'
        ratings = pd.read_sql_query(ratings_query, conn)
        movies = pd.read_sql_query(movies_query, conn)
        print("Successfully loaded data from database.")
        return ratings, movies
    except Error as e:
        print(f"Failed: {e}")
        return None, None
    finally:
        conn.close()

class EnhancedCF:
    def __init__(self, n_sim_user=20, n_rec_movie=10, pivot=0.75, n_factors=50, sim_method="Cosine"):
        self.n_sim_user = n_sim_user
        self.n_rec_movie = n_rec_movie
        self.pivot = pivot
        self.n_factors = n_factors
        self.sim_method = sim_method
        self.trainSet = {}
        self.testSet = {}
        self.user_list = None
        self.movie_list = None
        self.sim_matrix = None
        self.user_features = None
        self.movie_features = None
        self.sigma = None
        self.use_svd = False
        self.mean_ratings = None
        self.user_to_idx = None
        self.movie_to_idx = None
        self.user_ratings_matrix = None

    def get_dataset(self):
        rng = np.random.default_rng(seed=42)
        ratings, _ = load_data_from_db()
        if ratings is None:
            print("Failed to load ratings from database.")
            return

        count = 0
        for _, row in ratings.iterrows():
            user = str(row['userId'])
            movie = str(row['movieId'])
            rating = float(row['rating'])
            if rng.random() < self.pivot:
                self.trainSet.setdefault(user, {})[movie] = rating
                count += 1
            else:
                self.testSet.setdefault(user, {})[movie] = rating
            if count >= 500000:
                break

        print("Split training and test dataset successfully!")
        print(f"TrainSet size: {sum(len(movies) for movies in self.trainSet.values())}")
        print(f"TestSet size: {sum(len(movies) for movies in self.testSet.values())}")

    def _build_matrix(self):
        self.user_list = list(self.trainSet.keys())
        movie_set = set()
        for user in self.user_list:
            movie_set.update(self.trainSet[user].keys())
        self.movie_list = list(movie_set)

        self.user_to_idx = {u: i for i, u in enumerate(self.user_list)}
        self.movie_to_idx = {m: j for j, m in enumerate(self.movie_list)}
        
        data, rows, cols = [], [], []
        for user in self.user_list:
            for movie, rating in self.trainSet[user].items():
                rows.append(self.user_to_idx[user])
                cols.append(self.movie_to_idx[movie])
                data.append(rating)
        sparse_mat = csr_matrix((data, (rows, cols)), shape=(len(self.user_list), len(self.movie_list)))
        print(f"Sparse matrix shape: {sparse_mat.shape}")
        
        return sparse_mat

    def _compute_similarity(self, matrix, method="Cosine"):
        n_users = matrix.shape[0]
        sim_matrix = np.zeros((n_users, n_users))
        
        if method == "Cosine":
            return cosine_similarity(matrix)
            
        elif method == "Manhattan":
            dist_matrix = manhattan_distances(matrix)
            dist_matrix = np.where(dist_matrix == 0, 1e-10, dist_matrix)
            return 1.0 / (1.0 + dist_matrix)
            
        elif method in ["Pearson"]:
            dense_matrix = matrix.toarray() if hasattr(matrix, 'toarray') else matrix
            
            for i in range(n_users):
                for j in range(i, n_users):
                    if i == j:
                        sim_matrix[i, j] = 1.0
                        continue
                    u1_ratings = dense_matrix[i]
                    u2_ratings = dense_matrix[j]
                    mask = np.logical_and(u1_ratings > 0, u2_ratings > 0)
                    if np.sum(mask) < 2:
                        sim_matrix[i, j] = 0.0
                        sim_matrix[j, i] = 0.0
                        continue
                    
                    u1_common = u1_ratings[mask]
                    u2_common = u2_ratings[mask]
                    
                    if method == "Pearson":
                        try:
                            corr, _ = pearsonr(u1_common, u2_common)
                            if np.isnan(corr):
                                corr = 0.0
                        except:
                            corr = 0.0
                    sim = (corr + 1) / 2.0
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim
            
            return sim_matrix
        else:
            return cosine_similarity(matrix)

    def calc_user_sim_sparse(self):
        sparse_mat = self._build_matrix()
        self.sim_matrix = self._compute_similarity(sparse_mat, self.sim_method)
        self.use_svd = False

    def calc_user_sim_svd(self):
        sparse_mat = self._build_matrix()
        
        self.mean_ratings = np.zeros(sparse_mat.shape[0])
        for i in range(sparse_mat.shape[0]):
            row = sparse_mat.getrow(i).toarray().flatten()
            non_zero_indices = np.nonzero(row)[0]
            if len(non_zero_indices) > 0:
                self.mean_ratings[i] = np.mean(row[non_zero_indices])
            else:
                self.mean_ratings[i] = 3.0

        ratings_dense = sparse_mat.toarray()
        ratings_centered = ratings_dense.copy()
    
        for i in range(ratings_dense.shape[0]):
            zero_indices = ratings_dense[i] == 0
            ratings_centered[i, zero_indices] = 0
            non_zero_indices = ~zero_indices
            ratings_centered[i, non_zero_indices] -= self.mean_ratings[i]
        
        k = min(self.n_factors, min(ratings_centered.shape) - 1)
        U, sigma, Vt = svds(ratings_centered, k=k)
        
        U = np.flip(U, axis=1)
        sigma = np.flip(sigma)
        Vt = np.flip(Vt, axis=0)
        
        self.user_features = U
        self.sigma = sigma
        self.movie_features = Vt.T
        weighted_features = U.dot(np.diag(sigma))
        
        self.sim_matrix = self._compute_similarity(weighted_features, self.sim_method)

        self.use_svd = True

    def recommend_sparse(self, target_user):
        if target_user not in self.trainSet or target_user not in self.user_to_idx:
            print(f"User {target_user} is not in the training set")
            return []
        
        target_idx = self.user_to_idx[target_user]
        sim_vector = self.sim_matrix[target_idx]

        similar_idxs = np.argsort(-sim_vector)
        similar_user_idxs = [i for i in similar_idxs if self.user_list[i] != target_user][:self.n_sim_user]
        
        rank = {}
        watched_movies = set(self.trainSet[target_user].keys())
        for idx in similar_user_idxs:
            sim = self.sim_matrix[target_idx, idx]
            other_user = self.user_list[idx]
            for movie, rating in self.trainSet[other_user].items():
                if movie in watched_movies:
                    continue
                rank[movie] = rank.get(movie, 0) + sim
        recommended = sorted(rank.items(), key=lambda x: x[1], reverse=True)[:self.n_rec_movie]
        return recommended

    def recommend_svd(self, target_user):
        if target_user not in self.trainSet or target_user not in self.user_to_idx:
            print(f"User {target_user} is not in the training set")
            return []
            
        if self.user_features is None or self.movie_features is None or self.sigma is None or self.mean_ratings is None:
            print("SVD matrix not initialized")
            return []
        
        target_idx = self.user_to_idx[target_user]

        user_vec = self.user_features[target_idx]

        try:
            pred_ratings = self.mean_ratings[target_idx] + np.dot(user_vec * self.sigma, self.movie_features.T)
        except Exception as e:
            print(f"Error while calculating the expected ratings: {e}")
            return []
        
        watched_movies = set(self.trainSet[target_user].keys())
        watched_indices = [self.movie_to_idx[m] for m in watched_movies if m in self.movie_to_idx]
        candidate_indices = [i for i in range(len(self.movie_list)) if i not in watched_indices]
        candidate_ratings = [(self.movie_list[i], pred_ratings[i]) for i in candidate_indices]
        recommended = sorted(candidate_ratings, key=lambda x: x[1], reverse=True)[:self.n_rec_movie]
        return recommended

    def recommend(self, target_user):
        if self.use_svd:
            return self.recommend_svd(target_user)
        else:
            return self.recommend_sparse(target_user)

    def evaluate_model(self, k=None):
        if k is None:
            k = self.n_rec_movie

        precisions = []
        recalls = []
        f1_scores = []
        n_users_evaluated = 0
        for user in self.testSet:
            if user not in self.trainSet:
                continue
            ground_truth = set(self.testSet[user].keys())
            if not ground_truth:
                continue
                
            recommendations = self.recommend(user)
            rec_items = set([movie for movie, score in recommendations])
            if not rec_items:
                continue
                
            hit_count = len(rec_items.intersection(ground_truth))
            precision = hit_count / k
            recall = hit_count / len(ground_truth)
            
            precisions.append(precision)
            recalls.append(recall)
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
            f1_scores.append(f1)
            
            n_users_evaluated += 1

        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0
        avg_f1 = np.mean(f1_scores) if f1_scores else 0
        rec_items_all = set()
        sample_users = random.sample(list(self.trainSet.keys()), min(100, len(self.trainSet)))
        for user in sample_users:
            recs = self.recommend(user)
            for movie, score in recs:
                rec_items_all.add(movie)
        
        all_train_movies = set(self.movie_list)
        coverage = len(rec_items_all) / len(all_train_movies) if all_train_movies else 0

        metrics = {
            "Precision@{}".format(k): avg_precision,
            "Recall@{}".format(k): avg_recall,
            "F1@{}".format(k): avg_f1,
            "Coverage": coverage,
            "Model Type": "SVD" if self.use_svd else "Traditional CF"
        }
        
        print(f"Evaluated on {n_users_evaluated} users from test set.")
        return metrics

    def save_model(self, filepath=None):
        if filepath is None:
            model_type = "svd" if self.use_svd else "cf"
            method = self.sim_method.lower()
            filepath = f"model/{model_type}_{method}_model.pkl"
            
        state = {
            "n_sim_user": self.n_sim_user,
            "n_rec_movie": self.n_rec_movie,
            "pivot": self.pivot,
            "n_factors": self.n_factors,
            "sim_method": self.sim_method,
            "trainSet": self.trainSet,
            "testSet": self.testSet,
            "user_list": self.user_list,
            "movie_list": self.movie_list,
            "sim_matrix": self.sim_matrix,
            "use_svd": self.use_svd,
            "user_to_idx": self.user_to_idx,
            "movie_to_idx": self.movie_to_idx
        }
        
        if self.use_svd:
            state.update({
                "user_features": self.user_features,
                "movie_features": self.movie_features,
                "sigma": self.sigma,
                "mean_ratings": self.mean_ratings
            })
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath=None, model_type=None, sim_method=None):
        try:
            if filepath is None:
                if model_type is not None and sim_method is not None:
                    filepath = f"model/{model_type}_{sim_method.lower()}_model.pkl"
                    print("try to load model from", filepath)
                else:
                    print("try to load any valid model file")
                    model_files = []
                    for mt in ["svd", "cf"]:
                        for sm in ["cosine", "pearson", "manhattan"]:
                            file_path = f"model/{mt}_{sm}_model.pkl"
                            if os.path.exists(file_path):
                                model_files.append(file_path)
                    
                    if model_files:
                        filepath = model_files[0]
                    else:
                        print("Could not find any model file")
                        return False
                    
            if not os.path.exists(filepath):
                print(f"Model file {filepath} doesn't exist")
                return False
                
            with open(filepath, "rb") as f:
                state = pickle.load(f)
                
            required_keys = ["trainSet", "user_list", "sim_matrix"]
            for key in required_keys:
                if key not in state:
                    return False
                    
            self.n_sim_user = state.get("n_sim_user")
            self.n_rec_movie = state.get("n_rec_movie")
            self.pivot = state.get("pivot")
            self.n_factors = state.get("n_factors")
            self.trainSet = state.get("trainSet")
            self.testSet = state.get("testSet")
            self.user_list = state.get("user_list")
            self.movie_list = state.get("movie_list")
            self.sim_matrix = state.get("sim_matrix")
            self.use_svd = state.get("use_svd", False)
            self.user_to_idx = state.get("user_to_idx")
            self.movie_to_idx = state.get("movie_to_idx")
            self.sim_method = state.get("sim_method", "Cosine")
            
            if self.use_svd:
                svd_keys = ["user_features", "movie_features", "sigma", "mean_ratings"]
                for key in svd_keys:
                    if key not in state:
                        return False
                        
                self.user_features = state.get("user_features")
                self.movie_features = state.get("movie_features")
                self.sigma = state.get("sigma")
                self.mean_ratings = state.get("mean_ratings")
            return True
        except Exception as e:
            print(f"Error while loading the model: {e}")
            return False

if __name__ == '__main__':
    cf = EnhancedCF(n_sim_user=30, n_rec_movie=10, pivot=0.8, n_factors=35, sim_method="Cosine")
    
    if not cf.load_model():
        print("Fail to load model, will retrain the model")
        cf.get_dataset()
        
        model_type = input("Model type (1 = Collaborative Filtering, 2 = SVD decomposition) [Default value is 2]: ").strip()
        use_svd = True if not model_type or model_type == "2" else False
        
        print("\n Method to calculate similarity:")
        print("1: Cosine")
        print("2: Pearson")
        print("3: Manhattan")
        
        sim_choice = input("Choose method to calculate similarity (default value is 1): ").strip()
        sim_methods = {
            "1": "Cosine",
            "2": "Pearson", 
            "3": "Manhattan"
        }
        cf.sim_method = sim_methods.get(sim_choice, "Cosine")
        
        if use_svd:
            cf.calc_user_sim_svd()
        else:
            cf.calc_user_sim_sparse()
            
        cf.save_model()
    
    metrics = cf.evaluate_model()
    print("\n Model evaluation metric:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    print("\n Movie recommendation system")
    
    while True:
        target_user = input("\n Enter the user id for recommendation (enter q to quit): ").strip()
        if target_user.lower() == 'q':
            break
            
        try:
            target_user = str(float(target_user))
            recs = cf.recommend(target_user)
            
            if not recs:
                print(f"Cannot recommend movies for {target_user} ")
                continue
                
            print(f"\nRecommendations for user {target_user} :")
            for movie, score in recs:
                movie_details = query_movie_details(movie)
                if movie_details is not None and not movie_details.empty:
                    title = movie_details.iloc[0]['title']
                    genres = movie_details.iloc[0]['genres']
                else:
                    title, genres = "Unknown", "Unknown"
                print(f"Moving ID: {movie} | Title: {title} | Genre: {genres} | Score: {score:.4f}")
            
            user_ratings = query_user_ratings(target_user)
            if user_ratings is not None and not user_ratings.empty:
                top5 = user_ratings.sort_values(by="rating", ascending=False).head(5)
                for index, row in top5.iterrows():
                    movie_id = row["movieId"]
                    movie_details = query_movie_details(movie_id)
                    if movie_details is not None and not movie_details.empty:
                        title = movie_details.iloc[0]['title']
                        genres = movie_details.iloc[0]['genres']
                    else:
                        title, genres = "Unknown", "Unknown"
                    print(f"Movie ID: {movie_id} | Title: {title} | Genre: {genres} | Rating: {row['rating']}")
            else:
                print("No ratings for this user")
                
        except ValueError:
            print("Invalid user ID")
        except Exception as e:
            print(f"Error: {e}")