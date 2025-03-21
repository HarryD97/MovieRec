import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import math
import matplotlib.pyplot as plt
from datetime import datetime

if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)
print("Use device：", device)

model_dir = "model"
time_ncf_model_path = "model/time_ncf_model.pth"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

class TimeAwareMovieRatingDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        user = torch.tensor(row['user_idx'], dtype=torch.long)
        item = torch.tensor(row['movie_idx'], dtype=torch.long)
        rating = torch.tensor(row['rating'], dtype=torch.float)
        timestamp = torch.tensor(row['normalized_timestamp'], dtype=torch.float)
        return user, item, rating, timestamp

class TimeAwareNCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, time_embedding_dim=8):
        super(TimeAwareNCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.time_embedding_dim = time_embedding_dim
        
        self.fc1 = nn.Linear(embedding_dim * 2 + time_embedding_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.time_fc = nn.Linear(1, time_embedding_dim)
        
    def time_encoding(self, timestamp, k=10.0):
        time_input = timestamp.unsqueeze(1) if timestamp.dim() == 1 else timestamp
        recency_weight = torch.sigmoid(k * (time_input - 0.8))
        time_emb = F.relu(self.time_fc(time_input))
        time_emb = time_emb * recency_weight   
        return time_emb
        
    def forward(self, user, item, timestamp):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        time_emb = self.time_encoding(timestamp)

        x = torch.cat([user_emb, item_emb, time_emb], dim=-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        rating = torch.sigmoid(x) * 5.0
        return rating.squeeze()


def preprocess_data(ratings_path="ratings.csv", movies_path="movies.csv"):
    ratings_df = pd.read_csv(ratings_path)
    movies_df = pd.read_csv(movies_path)
    unique_users = ratings_df['userId'].unique()
    unique_items = ratings_df['movieId'].unique()
    user2idx = {user: idx for idx, user in enumerate(unique_users)}
    item2idx = {item: idx for idx, item in enumerate(unique_items)}
    idx2user = {idx: user for user, idx in user2idx.items()}
    idx2item = {idx: item for item, idx in item2idx.items()}

    ratings_df['user_idx'] = ratings_df['userId'].map(user2idx)
    ratings_df['movie_idx'] = ratings_df['movieId'].map(item2idx)

    min_timestamp = ratings_df['timestamp'].min()
    max_timestamp = ratings_df['timestamp'].max()
    time_range = max_timestamp - min_timestamp

    ratings_df['normalized_timestamp'] = (ratings_df['timestamp'] - min_timestamp) / time_range

    timestamp_info = {
        'min_timestamp': min_timestamp,
        'max_timestamp': max_timestamp,
        'time_range': time_range
    }
    
    num_users = len(unique_users)
    num_items = len(unique_items)
    print("Number of users：", num_users, "Number of movies：", num_items)
    print(f"Time range: {datetime.fromtimestamp(min_timestamp)} to {datetime.fromtimestamp(max_timestamp)}")
    
    ratings_df = ratings_df.sort_values('timestamp')
    train_size = int(len(ratings_df) * 0.8)
    train_df = ratings_df.iloc[:train_size]
    
    remaining_df = ratings_df.iloc[train_size:]
    val_df, test_df = train_test_split(remaining_df, test_size=0.5, random_state=42)
    train_min_time = datetime.fromtimestamp(train_df['timestamp'].min())
    train_max_time = datetime.fromtimestamp(train_df['timestamp'].max())
    val_min_time = datetime.fromtimestamp(val_df['timestamp'].min())
    val_max_time = datetime.fromtimestamp(val_df['timestamp'].max())
    test_min_time = datetime.fromtimestamp(test_df['timestamp'].min())
    test_max_time = datetime.fromtimestamp(test_df['timestamp'].max())

    return (train_df, val_df, test_df, movies_df, 
            user2idx, item2idx, idx2user, idx2item, 
            num_users, num_items, timestamp_info)

def train_time_aware_ncf(train_df, val_df, num_users, num_items, 
                         batch_size=256, num_epochs=20, learning_rate=0.001, load_existing=True):

    train_dataset = TimeAwareMovieRatingDataset(train_df)
    val_dataset = TimeAwareMovieRatingDataset(val_df)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TimeAwareNCF(num_users, num_items, embedding_dim=32, time_embedding_dim=8).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_epoch = 0
    if load_existing and os.path.exists(time_ncf_model_path):
        checkpoint = torch.load(time_ncf_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

        val_mse, val_rmse = evaluate_model(model, val_loader, criterion)
        
        return model, optimizer, start_epoch
    
    train_losses = []
    val_losses = []
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch + num_epochs}", ncols=100)
        
        for step, (user, item, rating, timestamp) in enumerate(progress_bar):
            user, item, rating, timestamp = user.to(device), item.to(device), rating.to(device), timestamp.to(device)
            
            optimizer.zero_grad()
            output = model(user, item, timestamp)
            loss = criterion(output, rating)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * user.size(0)
            progress_bar.set_postfix({'step': step, 'loss': loss.item()})
            
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        val_mse, val_rmse = evaluate_model(model, val_loader, criterion)
        val_losses.append(val_mse)
        
        print(f"Epoch {epoch+1}/{start_epoch + num_epochs}, Train Loss: {epoch_loss:.4f}, Val MSE: {val_mse:.4f}, Val RMSE: {val_rmse:.4f}")

        save_model(model, optimizer, epoch+1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Time-Aware NCF Training and Validation Loss')
    plt.legend()
    plt.savefig('time_ncf_loss_curve.png')
    plt.close()
    
    return model, optimizer, start_epoch + num_epochs

def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for user, item, rating, timestamp in dataloader:
            user = user.to(device)
            item = item.to(device)
            rating = rating.to(device)
            timestamp = timestamp.to(device)
            
            output = model(user, item, timestamp)
            loss = criterion(output, rating)
            batch_size = user.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    
    mse = total_loss / total_samples if total_samples > 0 else 0.0
    rmse = math.sqrt(mse)
    return mse, rmse

def save_model(model, optimizer, epoch, filepath=time_ncf_model_path):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")

def get_user_watching_history(user_id, ratings_df, movies_df):
    user_ratings = ratings_df[ratings_df['userId'] == user_id].sort_values('timestamp')
    user_movies = user_ratings.merge(movies_df, on='movieId')
    return user_movies

def recommend_movies(model, user_id, ratings_df, movies_df, user2idx, item2idx, timestamp_info, top_n=5, use_current_time=True):
    model.eval()
    
    if user_id not in user2idx:
        return None

    user_idx = user2idx[user_id]

    rated_movies = ratings_df[ratings_df["userId"] == user_id]
    if len(rated_movies) == 0:
        return None

    rated_items = rated_movies["movie_idx"].tolist()
    all_item_indices = list(range(len(item2idx)))
    candidate_indices = [i for i in all_item_indices if i not in rated_items]
    
    if len(candidate_indices) == 0:
        return None

    user_tensor = torch.tensor([user_idx] * len(candidate_indices), dtype=torch.long).to(device)
    item_tensor = torch.tensor(candidate_indices, dtype=torch.long).to(device)
    
    if use_current_time:
        current_unix_time = datetime.now().timestamp()
        current_unix_time = min(current_unix_time, timestamp_info['max_timestamp'])
        current_unix_time = max(current_unix_time, timestamp_info['min_timestamp'])
        normalized_time = (current_unix_time - timestamp_info['min_timestamp']) / timestamp_info['time_range']
    else:
        normalized_time = 1.0
    
    time_tensor = torch.tensor([normalized_time] * len(candidate_indices), dtype=torch.float).to(device)
    
    with torch.no_grad():
        preds = model(user_tensor, item_tensor, time_tensor)
    
    preds = preds.cpu().numpy()
    
    user_history = get_user_watching_history(user_id, ratings_df, movies_df)
    
    if len(user_history) >= 3:
        recent_movies = user_history.sort_values('timestamp', ascending=False).head(5)
        genre_preferences = {}
        for _, movie in recent_movies.iterrows():
            genres = movie['genres'].split('|')
            for genre in genres:
                if genre in genre_preferences:
                    genre_preferences[genre] += 1
                else:
                    genre_preferences[genre] = 1

        total_genres = sum(genre_preferences.values())
        for genre in genre_preferences:
            genre_preferences[genre] /= total_genres

        recency_weight = 0.3
        for i, idx in enumerate(candidate_indices):
            movie_id = None
            for mid, midx in item2idx.items():
                if midx == idx:
                    movie_id = mid
                    break
            
            if movie_id is None:
                continue

            movie_info = movies_df[movies_df['movieId'] == movie_id]
            if len(movie_info) == 0:
                continue

            movie_genres = movie_info.iloc[0]['genres'].split('|')
            genre_score = sum([genre_preferences.get(genre, 0) for genre in movie_genres])

            preds[i] = (1 - recency_weight) * preds[i] + recency_weight * genre_score * 5.0

    top_indices = np.argsort(preds)[::-1][:top_n]
    recommended_item_indices = [candidate_indices[i] for i in top_indices]

    idx2item = {idx: item for item, idx in item2idx.items()}
    recommended_movie_ids = [idx2item[i] for i in recommended_item_indices]

    recommendations = movies_df[movies_df["movieId"].isin(recommended_movie_ids)].copy()

    movie_id_to_pred = {}
    for i, idx in enumerate(top_indices):
        orig_idx = candidate_indices[idx]
        movie_id = idx2item[orig_idx]
        movie_id_to_pred[movie_id] = preds[idx]
    
    recommendations['predicted_rating'] = recommendations['movieId'].map(lambda x: movie_id_to_pred.get(x, 0))
    
    return recommendations.sort_values('predicted_rating', ascending=False)

def main():
    train_df, val_df, test_df, movies_df, user2idx, item2idx, idx2user, idx2item, num_users, num_items, timestamp_info = preprocess_data('./dataset/ratings.csv', './dataset/movies.csv')

    model, optimizer, trained_epochs = train_time_aware_ncf(
        train_df, val_df, num_users, num_items, 
        batch_size=256, num_epochs=20, learning_rate=0.001, 
        load_existing=True
    )
    
    test_dataset = TimeAwareMovieRatingDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    criterion = nn.MSELoss()
    
    test_mse, test_rmse = evaluate_model(model, test_loader, criterion)

    sample_users = [1, 5, 10]  
    for user_id in sample_users:
        user_history = get_user_watching_history(user_id, train_df, movies_df)
        
        if len(user_history) == 0:
            continue

        recent_history = user_history.sort_values('timestamp', ascending=False).head(5)
        for i, (_, movie) in enumerate(recent_history.iterrows(), 1):
            watch_time = datetime.fromtimestamp(movie['timestamp'])
            print(f"{i}. {movie['title']} ({movie['genres']}) - Rating: {movie['rating']}")
            print(f"Watching time: {watch_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        recommendations = recommend_movies(
            model, user_id, train_df, movies_df, 
            user2idx, item2idx, timestamp_info, top_n=5, use_current_time=True
        )
        
        if recommendations is not None:
            print(f"\n Movies recommended for {user_id}:")
            for i, (_, movie) in enumerate(recommendations.iterrows(), 1):
                print(f"{i}. {movie['title']} ({movie['genres']}) - expected score: {movie['predicted_rating']:.2f}")

if __name__ == "__main__":
    main()