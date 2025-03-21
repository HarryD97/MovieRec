import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datetime import datetime

# 检查GPU是否可用，并设置运行设备
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)
print("使用设备：", device)

# 模型保存路径
model_dir = "model"
model_path = "model/ncf_model.pth"
time_aware_model_path = "model/time_ncf_model.pth"

# 确保模型目录存在
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# -----------------------------
# 数据加载与预处理
# -----------------------------
def load_and_preprocess_data(rating_path="ratings.csv", movie_path='movies.csv'):
    """加载和预处理数据"""
    print("加载并预处理数据...")
    
    # 读取数据
    ratings_df = pd.read_csv(rating_path)
    movies_df = pd.read_csv(movie_path)
    
    # 构造用户和电影ID到索引的映射
    unique_users = ratings_df['userId'].unique()
    unique_items = ratings_df['movieId'].unique()
    user2idx = {user: idx for idx, user in enumerate(unique_users)}
    item2idx = {item: idx for idx, item in enumerate(unique_items)}
    idx2user = {idx: user for user, idx in user2idx.items()}
    idx2item = {idx: item for item, idx in item2idx.items()}
    
    # 为了方便后续计算，将原始的 userId 和 movieId 转换为索引
    ratings_df['user_idx'] = ratings_df['userId'].map(user2idx)
    ratings_df['movie_idx'] = ratings_df['movieId'].map(item2idx)
    
    # 归一化时间戳
    max_timestamp = ratings_df['timestamp'].max()
    ratings_df['normalized_timestamp'] = ratings_df['timestamp'] / max_timestamp
    
    num_users = len(unique_users)
    num_items = len(unique_items)
    print("用户数量：", num_users, "电影数量：", num_items)
    
    return ratings_df, movies_df, user2idx, item2idx, idx2user, idx2item, num_users, num_items, max_timestamp

# -----------------------------
# 数据集划分
# -----------------------------
def split_data(ratings_df):
    """将数据集划分为训练集、验证集和测试集"""
    print("划分数据集...")
    
    # 划分数据：训练集 80%，验证集 10%，测试集 10%
    train_val_df, test_df = train_test_split(ratings_df, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1111, random_state=42)  # 0.1111≈10%/90%

    print("训练集大小：", len(train_df))
    print("验证集大小：", len(val_df))
    print("测试集大小：", len(test_df))
    
    return train_df, val_df, test_df

# -----------------------------
# 自定义 Dataset
# -----------------------------
class MovieRatingDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        user = torch.tensor(row['user_idx'], dtype=torch.long)
        item = torch.tensor(row['movie_idx'], dtype=torch.long)
        rating = torch.tensor(row['rating'], dtype=torch.float)
        return user, item, rating

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

# -----------------------------
# 定义神经协同过滤模型（NCF）
# -----------------------------
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        x = torch.cat([user_emb, item_emb], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        rating = torch.sigmoid(x) * 5.0
        return rating.squeeze()

# -----------------------------
# 定义时间感知的神经协同过滤模型
# -----------------------------
class TimeAwareNCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, time_embedding_dim=8):
        super(TimeAwareNCF, self).__init__()
        # 用户和电影的嵌入
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 时间嵌入
        self.time_embedding_dim = time_embedding_dim
        
        # 全连接层，增加了时间嵌入的维度
        self.fc1 = nn.Linear(embedding_dim * 2 + time_embedding_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        # 时间编码转换
        self.time_fc = nn.Linear(1, time_embedding_dim)
        
    def time_encoding(self, timestamp):
        """将时间戳转换为嵌入向量"""
        # 调整形状为 [batch_size, 1]
        time_input = timestamp.unsqueeze(1) if timestamp.dim() == 1 else timestamp
        
        # 通过全连接层将时间转换为嵌入
        time_emb = F.relu(self.time_fc(time_input))
        return time_emb
        
    def forward(self, user, item, timestamp):
        # 获取用户和电影嵌入
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        
        # 获取时间嵌入
        time_emb = self.time_encoding(timestamp)
        
        # 将三种嵌入连接在一起
        x = torch.cat([user_emb, item_emb, time_emb], dim=-1)
        
        # 通过全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # 输出预测评分（0-5分）
        rating = torch.sigmoid(x) * 5.0
        return rating.squeeze()

# -----------------------------
# 模型训练和评估函数
# -----------------------------
def train_ncf_model(train_loader, val_loader, num_users, num_items, num_epochs=5, learning_rate=0.001, load_existing=True):
    """训练标准NCF模型"""
    model = NCF(num_users, num_items, embedding_dim=32).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 尝试加载已有模型
    start_epoch = 0
    if load_existing and os.path.exists(model_path):
        print(f"找到已有模型，正在加载...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"成功加载模型，已训练 {start_epoch} 个轮次")
        
        # 评估加载的模型
        val_mse, val_rmse = evaluate_model(model, val_loader, criterion, device)
        print(f"加载的模型在验证集上表现: MSE = {val_mse:.4f}, RMSE = {val_rmse:.4f}")
        
        return model, optimizer, start_epoch
    
    # 如果没有找到已有模型或不加载已有模型，从头训练
    print("开始训练NCF模型...")
    
    train_losses = []
    val_losses = []
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch + num_epochs}", ncols=100)
        
        for step, (user, item, rating) in enumerate(progress_bar):
            user, item, rating = user.to(device), item.to(device), rating.to(device)
            
            optimizer.zero_grad()
            output = model(user, item)
            loss = criterion(output, rating)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * user.size(0)
            progress_bar.set_postfix({'step': step, 'loss': loss.item()})
            
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # 验证阶段
        val_mse, val_rmse = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_mse)
        
        print(f"Epoch {epoch+1}/{start_epoch + num_epochs}, Train Loss: {epoch_loss:.4f}, Val MSE: {val_mse:.4f}, Val RMSE: {val_rmse:.4f}")
        
        # 每个epoch保存一次模型
        save_model(model, optimizer, epoch+1, model_path)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('ncf_loss_curve.png')
    plt.close()
    
    return model, optimizer, start_epoch + num_epochs

def train_time_aware_ncf_model(train_loader, val_loader, num_users, num_items, num_epochs=5, learning_rate=0.001, load_existing=True):
    """训练时间感知NCF模型"""
    model = TimeAwareNCF(num_users, num_items, embedding_dim=32, time_embedding_dim=8).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 尝试加载已有模型
    start_epoch = 0
    if load_existing and os.path.exists(time_aware_model_path):
        print(f"找到已有时间感知模型，正在加载...")
        checkpoint = torch.load(time_aware_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"成功加载模型，已训练 {start_epoch} 个轮次")
        
        # 评估加载的模型
        val_mse, val_rmse = evaluate_time_aware_model(model, val_loader, criterion, device)
        print(f"加载的时间感知模型在验证集上表现: MSE = {val_mse:.4f}, RMSE = {val_rmse:.4f}")
        
        return model, optimizer, start_epoch
    
    # 如果没有找到已有模型或不加载已有模型，从头训练
    print("开始训练时间感知NCF模型...")
    
    train_losses = []
    val_losses = []
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # 训练阶段
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
        
        # 验证阶段
        val_mse, val_rmse = evaluate_time_aware_model(model, val_loader, criterion, device)
        val_losses.append(val_mse)
        
        print(f"Epoch {epoch+1}/{start_epoch + num_epochs}, Train Loss: {epoch_loss:.4f}, Val MSE: {val_mse:.4f}, Val RMSE: {val_rmse:.4f}")
        
        # 每个epoch保存一次模型
        save_model(model, optimizer, epoch+1, time_aware_model_path)
    
    # 绘制损失曲线
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

def evaluate_model(model, dataloader, criterion, device):
    """评估标准NCF模型"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for user, item, rating in dataloader:
            user = user.to(device)
            item = item.to(device)
            rating = rating.to(device)
            
            output = model(user, item)
            loss = criterion(output, rating)
            batch_size = user.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    
    mse = total_loss / total_samples if total_samples > 0 else 0.0
    rmse = math.sqrt(mse)
    return mse, rmse

def evaluate_time_aware_model(model, dataloader, criterion, device):
    """评估时间感知NCF模型"""
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

def save_model(model, optimizer, epoch, filepath):
    """保存模型"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)
    print(f"模型已保存到 {filepath}")

# -----------------------------
# 基于观影历史的推荐函数
# -----------------------------
def get_user_watching_history(user_id, ratings_df, movies_df):
    """获取用户的观影历史"""
    user_ratings = ratings_df[ratings_df['userId'] == user_id].sort_values('timestamp')
    user_movies = user_ratings.merge(movies_df, on='movieId')
    return user_movies

def recommend_movies_with_history(model, user_id, ratings_df, movies_df, user2idx, item2idx, top_n=5, time_aware=False, recency_weight=0.3):
    """基于用户的观影历史推荐电影
    
    Args:
        model: 训练好的模型
        user_id: 用户ID
        ratings_df: 评分数据
        movies_df: 电影数据
        user2idx: 用户ID到索引的映射
        item2idx: 电影ID到索引的映射
        top_n: 推荐电影数量
        time_aware: 是否使用时间感知模型
        recency_weight: 最近观看的电影的权重
        
    Returns:
        推荐的电影列表
    """
    model.eval()
    
    # 检查用户是否存在
    if user_id not in user2idx:
        print(f"用户 {user_id} 不存在")
        return None
    
    # 获取用户索引
    user_idx = user2idx[user_id]
    
    # 获取用户已观看电影
    rated_movies = ratings_df[ratings_df["userId"] == user_id]
    if len(rated_movies) == 0:
        print(f"用户 {user_id} 没有观看记录")
        return None
    
    # 获取用户未观看电影的索引
    rated_items = rated_movies["movie_idx"].tolist()
    all_item_indices = list(range(len(item2idx)))
    candidate_indices = [i for i in all_item_indices if i not in rated_items]
    
    if len(candidate_indices) == 0:
        print("用户已经观看所有电影，无可推荐项目。")
        return None
    
    # 准备模型输入
    user_tensor = torch.tensor([user_idx] * len(candidate_indices), dtype=torch.long).to(device)
    item_tensor = torch.tensor(candidate_indices, dtype=torch.long).to(device)
    
    # 当前时间戳（归一化后）
    current_time = 1.0  # 假设1.0表示当前时间（归一化后的最大值）
    time_tensor = torch.tensor([current_time] * len(candidate_indices), dtype=torch.float).to(device)
    
    # 预测评分
    with torch.no_grad():
        if time_aware:
            preds = model(user_tensor, item_tensor, time_tensor)
        else:
            preds = model(user_tensor, item_tensor)
    
    preds = preds.cpu().numpy()
    
    # 获取用户观影历史并按时间排序
    user_history = get_user_watching_history(user_id, ratings_df, movies_df)
    
    # 提取用户最近观看的电影的类型偏好
    if len(user_history) >= 3:  # 至少有3部电影才计算类型偏好
        # 获取用户最近观看的N部电影
        recent_movies = user_history.sort_values('timestamp', ascending=False).head(5)
        
        # 提取这些电影的类型
        genre_preferences = {}
        for _, movie in recent_movies.iterrows():
            genres = movie['genres'].split('|')
            for genre in genres:
                if genre in genre_preferences:
                    genre_preferences[genre] += 1
                else:
                    genre_preferences[genre] = 1
        
        # 标准化类型偏好
        total_genres = sum(genre_preferences.values())
        for genre in genre_preferences:
            genre_preferences[genre] /= total_genres
        
        # 根据候选电影的类型调整预测分数
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
            
            # 调整预测分数，考虑最近的类型偏好
            preds[i] = (1 - recency_weight) * preds[i] + recency_weight * genre_score * 5.0  # 缩放到0-5分范围
    
    # 按调整后的分数排序
    top_indices = np.argsort(preds)[::-1][:top_n]
    recommended_item_indices = [candidate_indices[i] for i in top_indices]
    
    # 将索引转换回电影ID
    idx2item = {idx: item for item, idx in item2idx.items()}
    recommended_movie_ids = [idx2item[i] for i in recommended_item_indices]
    
    # 获取推荐电影的详细信息
    recommendations = movies_df[movies_df["movieId"].isin(recommended_movie_ids)].copy()
    
    # 添加预测评分 - 修复此处的索引错误
    # 创建一个电影ID到预测分数的映射
    movie_id_to_pred = {}
    for i, idx in enumerate(top_indices):
        movie_id = idx2item[candidate_indices[idx]]
        movie_id_to_pred[movie_id] = preds[idx]
    
    # 为每部推荐电影添加预测评分
    recommendations['predicted_rating'] = recommendations['movieId'].map(lambda x: movie_id_to_pred.get(x, 0))
    
    return recommendations.sort_values('predicted_rating', ascending=False)

def show_user_history_and_recommend(user_id, ratings_df, movies_df, user2idx, item2idx, time_aware=False, top_n=5):
    """展示用户的观影历史，并给出推荐"""
    # 获取用户观影历史
    user_history = get_user_watching_history(user_id, ratings_df, movies_df)
    
    if len(user_history) == 0:
        print(f"用户 {user_id} 没有观影记录")
        return
    
    # 显示用户最近观看的5部电影
    print(f"\n用户 {user_id} 的最近观影历史:")
    recent_history = user_history.sort_values('timestamp', ascending=False).head(5)
    for i, (_, movie) in enumerate(recent_history.iterrows(), 1):
        print(f"{i}. {movie['title']} ({movie['genres']}) - 评分: {movie['rating']}")
        watch_time = datetime.fromtimestamp(movie['timestamp'])
        print(f"   观看时间: {watch_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 尝试加载模型并推荐
    if time_aware:
        model_file = time_aware_model_path
        model = TimeAwareNCF(len(user2idx), len(item2idx), embedding_dim=32, time_embedding_dim=8).to(device)
    else:
        model_file = model_path
        model = NCF(len(user2idx), len(item2idx), embedding_dim=32).to(device)
    
    if os.path.exists(model_file):
        checkpoint = torch.load(model_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n使用{'时间感知' if time_aware else '标准'} NCF模型推荐:")
        
        recommendations = recommend_movies_with_history(
            model, user_id, ratings_df, movies_df, user2idx, item2idx, 
            top_n=top_n, time_aware=time_aware
        )
        
        if recommendations is not None:
            print(f"\n对用户 {user_id} 的电影推荐:")
            for i, (_, movie) in enumerate(recommendations.iterrows(), 1):
                print(f"{i}. {movie['title']} ({movie['genres']}) - 预测评分: {movie['predicted_rating']:.2f}")
        return recommendations
    else:
        print(f"\n未找到{'时间感知' if time_aware else '标准'} NCF模型，请先训练模型")
    return None


def show_user_history_and_recommend_debug(user_id, ratings_df, movies_df, user2idx, item2idx, model, time_aware=False, top_n=5):
    """展示用户的观影历史，并给出推荐"""
    # 获取用户观影历史
    user_history = get_user_watching_history(user_id, ratings_df, movies_df)

    if len(user_history) == 0:
        # print(f"用户 {user_id} 没有观影记录")
        return

    # 显示用户最近观看的5部电影
    # print(f"\n用户 {user_id} 的最近观影历史:")
    # recent_history = user_history.sort_values('timestamp', ascending=False).head(5)
    # for i, (_, movie) in enumerate(recent_history.iterrows(), 1):
    #     print(f"{i}. {movie['title']} ({movie['genres']}) - 评分: {movie['rating']}")
    #     watch_time = datetime.fromtimestamp(movie['timestamp'])
    #     print(f"   观看时间: {watch_time.strftime('%Y-%m-%d %H:%M:%S')}")

    recommendations = recommend_movies_with_history(
        model, user_id, ratings_df, movies_df, user2idx, item2idx,
        top_n=top_n, time_aware=time_aware
    )

    # if recommendations is not None:
        # print(f"\n对用户 {user_id} 的电影推荐:")
        # for i, (_, movie) in enumerate(recommendations.iterrows(), 1):
        #     print(f"{i}. {movie['title']} ({movie['genres']}) - 预测评分: {movie['predicted_rating']:.2f}")
    return recommendations

# -----------------------------
# 主函数
# -----------------------------
def main():
    # 加载并预处理数据
    ratings_df, movies_df, user2idx, item2idx, idx2user, idx2item, num_users, num_items, max_timestamp = load_and_preprocess_data()
    
    # 划分数据
    train_df, val_df, test_df = split_data(ratings_df)
    
    # 创建标准DataLoader
    train_dataset = MovieRatingDataset(train_df)
    val_dataset = MovieRatingDataset(val_df)
    test_dataset = MovieRatingDataset(test_df)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # 创建时间感知DataLoader
    time_train_dataset = TimeAwareMovieRatingDataset(train_df)
    time_val_dataset = TimeAwareMovieRatingDataset(val_df)
    time_test_dataset = TimeAwareMovieRatingDataset(test_df)
    
    time_train_loader = DataLoader(time_train_dataset, batch_size=256, shuffle=True)
    time_val_loader = DataLoader(time_val_dataset, batch_size=256, shuffle=False)
    time_test_loader = DataLoader(time_test_dataset, batch_size=256, shuffle=False)
    
    # 训练标准NCF模型或加载已有模型
    standard_model, standard_optimizer, standard_epochs = train_ncf_model(
        train_loader, val_loader, num_users, num_items, 
        num_epochs=30, learning_rate=0.001, load_existing=True
    )
    
    # 训练时间感知NCF模型或加载已有模型
    time_model, time_optimizer, time_epochs = train_time_aware_ncf_model(
        time_train_loader, time_val_loader, num_users, num_items, 
        num_epochs=30, learning_rate=0.001, load_existing=True
    )
    
    # 评估模型
    criterion = nn.MSELoss()
    print("\n评估标准NCF模型在测试集上的表现:")
    test_mse, test_rmse = evaluate_model(standard_model, test_loader, criterion, device)
    print(f"测试集性能: MSE = {test_mse:.4f}, RMSE = {test_rmse:.4f}")
    
    print("\n评估时间感知NCF模型在测试集上的表现:")
    time_test_mse, time_test_rmse = evaluate_time_aware_model(time_model, time_test_loader, criterion, device)
    print(f"测试集性能: MSE = {time_test_mse:.4f}, RMSE = {time_test_rmse:.4f}")
    
    # 为几个样本用户生成推荐
    sample_users = ratings_df['userId'].sample(3).tolist()
    for user_id in sample_users:
        # 使用标准NCF模型推荐
        print("\n" + "-"*50)
        print(f"用户 {user_id} 的推荐:")
        show_user_history_and_recommend(user_id, ratings_df, movies_df, user2idx, item2idx, time_aware=False)
        
        # 使用时间感知NCF模型推荐
        print("\n使用时间感知模型:")
        show_user_history_and_recommend(user_id, ratings_df, movies_df, user2idx, item2idx, time_aware=True)

if __name__ == "__main__":
    main()