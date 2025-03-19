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
time_ncf_model_path = "model/time_ncf_model.pth"

# 确保模型目录存在
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# -----------------------------
# 数据加载与预处理
# -----------------------------
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
        
    def time_encoding(self, timestamp, k=10.0):
        """
        时间编码函数：让更近的时间有更高的权重
        
        参数:
        - timestamp: 归一化的时间戳
        - k: 控制权重曲线陡峭程度的参数
        
        返回:
        - 时间嵌入向量，近期时间戳有更高的权重
        """
        time_input = timestamp.unsqueeze(1) if timestamp.dim() == 1 else timestamp
        
        # 因为timestamp已经归一化到0-1范围，且值越大表示越近
        # 直接使用sigmoid函数增强时间的新近性权重
        # 将timestamp缩放到更合适的范围，让sigmoid在近期时间有更明显的区分度
        recency_weight = torch.sigmoid(k * (time_input - 0.8))
        
        # 计算时间嵌入
        time_emb = F.relu(self.time_fc(time_input))
        
        # 让时间嵌入乘以新近性权重
        time_emb = time_emb * recency_weight
        
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
# 数据预处理函数
# -----------------------------
def preprocess_data(ratings_path="ratings.csv", movies_path="movies.csv"):
    """加载和预处理数据"""
    print("加载并预处理数据...")
    
    # 读取数据
    ratings_df = pd.read_csv(ratings_path)
    movies_df = pd.read_csv(movies_path)
    
    # 构造用户和电影ID到索引的映射
    unique_users = ratings_df['userId'].unique()
    unique_items = ratings_df['movieId'].unique()
    user2idx = {user: idx for idx, user in enumerate(unique_users)}
    item2idx = {item: idx for idx, item in enumerate(unique_items)}
    idx2user = {idx: user for user, idx in user2idx.items()}
    idx2item = {idx: item for item, idx in item2idx.items()}
    
    # 将原始的 userId 和 movieId 转换为索引
    ratings_df['user_idx'] = ratings_df['userId'].map(user2idx)
    ratings_df['movie_idx'] = ratings_df['movieId'].map(item2idx)
    
    # 时间戳分析
    min_timestamp = ratings_df['timestamp'].min()
    max_timestamp = ratings_df['timestamp'].max()
    time_range = max_timestamp - min_timestamp
    
    # 将时间戳归一化到[0,1]区间，值越大表示时间越近
    ratings_df['normalized_timestamp'] = (ratings_df['timestamp'] - min_timestamp) / time_range
    
    # 保存最大和最小时间戳，用于后续推荐
    timestamp_info = {
        'min_timestamp': min_timestamp,
        'max_timestamp': max_timestamp,
        'time_range': time_range
    }
    
    num_users = len(unique_users)
    num_items = len(unique_items)
    print("用户数量：", num_users, "电影数量：", num_items)
    print(f"时间范围: {datetime.fromtimestamp(min_timestamp)} 到 {datetime.fromtimestamp(max_timestamp)}")
    
    # 按时间排序评分数据
    ratings_df = ratings_df.sort_values('timestamp')
    
    # 将最早的80%数据作为训练集
    train_size = int(len(ratings_df) * 0.8)
    train_df = ratings_df.iloc[:train_size]
    
    # 剩下的20%随机分配给测试集和验证集
    remaining_df = ratings_df.iloc[train_size:]
    val_df, test_df = train_test_split(remaining_df, test_size=0.5, random_state=42)
    
    print("训练集大小（最早的80%数据）：", len(train_df))
    print("验证集大小（剩余数据的随机50%）：", len(val_df))
    print("测试集大小（剩余数据的随机50%）：", len(test_df))
    
    # 时间分析
    train_min_time = datetime.fromtimestamp(train_df['timestamp'].min())
    train_max_time = datetime.fromtimestamp(train_df['timestamp'].max())
    val_min_time = datetime.fromtimestamp(val_df['timestamp'].min())
    val_max_time = datetime.fromtimestamp(val_df['timestamp'].max())
    test_min_time = datetime.fromtimestamp(test_df['timestamp'].min())
    test_max_time = datetime.fromtimestamp(test_df['timestamp'].max())
    
    print(f"训练集时间范围: {train_min_time} 到 {train_max_time}")
    print(f"验证集时间范围: {val_min_time} 到 {val_max_time}")
    print(f"测试集时间范围: {test_min_time} 到 {test_max_time}")
    
    return (train_df, val_df, test_df, movies_df, 
            user2idx, item2idx, idx2user, idx2item, 
            num_users, num_items, timestamp_info)

# -----------------------------
# 训练和评估函数
# -----------------------------
def train_time_aware_ncf(train_df, val_df, num_users, num_items, 
                         batch_size=256, num_epochs=20, learning_rate=0.001, load_existing=True):
    """训练时间感知NCF模型"""
    # 创建数据加载器
    train_dataset = TimeAwareMovieRatingDataset(train_df)
    val_dataset = TimeAwareMovieRatingDataset(val_df)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型、损失函数和优化器
    model = TimeAwareNCF(num_users, num_items, embedding_dim=32, time_embedding_dim=8).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 尝试加载已有模型
    start_epoch = 0
    if load_existing and os.path.exists(time_ncf_model_path):
        print(f"找到已有时间感知模型，正在加载...")
        checkpoint = torch.load(time_ncf_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"成功加载模型，已训练 {start_epoch} 个轮次")
        
        # 评估加载的模型
        val_mse, val_rmse = evaluate_model(model, val_loader, criterion)
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
        val_mse, val_rmse = evaluate_model(model, val_loader, criterion)
        val_losses.append(val_mse)
        
        print(f"Epoch {epoch+1}/{start_epoch + num_epochs}, Train Loss: {epoch_loss:.4f}, Val MSE: {val_mse:.4f}, Val RMSE: {val_rmse:.4f}")
        
        # 每个epoch保存一次模型
        save_model(model, optimizer, epoch+1)
    
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

def evaluate_model(model, dataloader, criterion):
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

def save_model(model, optimizer, epoch, filepath=time_ncf_model_path):
    """保存模型"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)
    print(f"模型已保存到 {filepath}")

# -----------------------------
# 推荐函数
# -----------------------------
def get_user_watching_history(user_id, ratings_df, movies_df):
    """获取用户的观影历史"""
    user_ratings = ratings_df[ratings_df['userId'] == user_id].sort_values('timestamp')
    user_movies = user_ratings.merge(movies_df, on='movieId')
    return user_movies

def recommend_movies(model, user_id, ratings_df, movies_df, user2idx, item2idx, timestamp_info, top_n=5, use_current_time=True):
    """使用时间感知NCF模型为用户推荐电影"""
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
    
    # 使用当前时间或最大时间作为预测时间点
    if use_current_time:
        # 使用当前时间（归一化到与训练数据相同的范围）
        current_unix_time = datetime.now().timestamp()
        # 确保时间戳在合理范围内
        current_unix_time = min(current_unix_time, timestamp_info['max_timestamp'])
        current_unix_time = max(current_unix_time, timestamp_info['min_timestamp'])
        # 归一化
        normalized_time = (current_unix_time - timestamp_info['min_timestamp']) / timestamp_info['time_range']
    else:
        # 使用最大时间（最近的数据点）
        normalized_time = 1.0
    
    time_tensor = torch.tensor([normalized_time] * len(candidate_indices), dtype=torch.float).to(device)
    
    # 预测评分
    with torch.no_grad():
        preds = model(user_tensor, item_tensor, time_tensor)
    
    preds = preds.cpu().numpy()
    
    # 获取用户观影历史
    user_history = get_user_watching_history(user_id, ratings_df, movies_df)
    
    # 根据用户最近观看电影的类型调整预测分数
    if len(user_history) >= 3:
        # 获取用户最近观看的5部电影
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
        
        # 调整预测分数，考虑类型偏好
        recency_weight = 0.3  # 类型相似度的权重
        for i, idx in enumerate(candidate_indices):
            # 获取电影ID
            movie_id = None
            for mid, midx in item2idx.items():
                if midx == idx:
                    movie_id = mid
                    break
            
            if movie_id is None:
                continue
                
            # 获取电影信息
            movie_info = movies_df[movies_df['movieId'] == movie_id]
            if len(movie_info) == 0:
                continue
                
            # 计算类型相似度
            movie_genres = movie_info.iloc[0]['genres'].split('|')
            genre_score = sum([genre_preferences.get(genre, 0) for genre in movie_genres])
            
            # 调整预测分数
            preds[i] = (1 - recency_weight) * preds[i] + recency_weight * genre_score * 5.0
    
    # 按预测分数排序，选出top_n
    top_indices = np.argsort(preds)[::-1][:top_n]
    recommended_item_indices = [candidate_indices[i] for i in top_indices]
    
    # 将索引转换回电影ID
    idx2item = {idx: item for item, idx in item2idx.items()}
    recommended_movie_ids = [idx2item[i] for i in recommended_item_indices]
    
    # 获取推荐电影详情
    recommendations = movies_df[movies_df["movieId"].isin(recommended_movie_ids)].copy()
    
    # 添加预测评分
    movie_id_to_pred = {}
    for i, idx in enumerate(top_indices):
        orig_idx = candidate_indices[idx]
        movie_id = idx2item[orig_idx]
        movie_id_to_pred[movie_id] = preds[idx]
    
    recommendations['predicted_rating'] = recommendations['movieId'].map(lambda x: movie_id_to_pred.get(x, 0))
    
    return recommendations.sort_values('predicted_rating', ascending=False)

# -----------------------------
# 主函数：运行示例
# -----------------------------
def main():
    # 预处理数据
    train_df, val_df, test_df, movies_df, user2idx, item2idx, idx2user, idx2item, num_users, num_items, timestamp_info = preprocess_data('./dataset/ratings.csv', './dataset/movies.csv')
    
    # 训练模型（或加载已有模型）
    model, optimizer, trained_epochs = train_time_aware_ncf(
        train_df, val_df, num_users, num_items, 
        batch_size=256, num_epochs=20, learning_rate=0.001, 
        load_existing=True
    )
    
    # 在测试集上评估模型
    test_dataset = TimeAwareMovieRatingDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    criterion = nn.MSELoss()
    
    test_mse, test_rmse = evaluate_model(model, test_loader, criterion)
    print(f"测试集性能: MSE = {test_mse:.4f}, RMSE = {test_rmse:.4f}")
    
    # 为样本用户生成推荐
    sample_users = [1, 5, 10]  # 示例用户
    print("\n为样本用户生成电影推荐:")
    
    for user_id in sample_users:
        print(f"\n用户 {user_id} 的推荐:")
        
        # 获取用户观影历史
        user_history = get_user_watching_history(user_id, train_df, movies_df)
        
        if len(user_history) == 0:
            print(f"用户 {user_id} 没有观影记录")
            continue
            
        print(f"用户 {user_id} 的最近观影历史 (前5部):")
        recent_history = user_history.sort_values('timestamp', ascending=False).head(5)
        for i, (_, movie) in enumerate(recent_history.iterrows(), 1):
            watch_time = datetime.fromtimestamp(movie['timestamp'])
            print(f"{i}. {movie['title']} ({movie['genres']}) - 评分: {movie['rating']}")
            print(f"   观看时间: {watch_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 生成推荐
        recommendations = recommend_movies(
            model, user_id, train_df, movies_df, 
            user2idx, item2idx, timestamp_info, top_n=5, use_current_time=True
        )
        
        if recommendations is not None:
            print(f"\n为用户 {user_id} 推荐的电影:")
            for i, (_, movie) in enumerate(recommendations.iterrows(), 1):
                print(f"{i}. {movie['title']} ({movie['genres']}) - 预测评分: {movie['predicted_rating']:.2f}")

if __name__ == "__main__":
    main()