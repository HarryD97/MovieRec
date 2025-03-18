#!/usr/bin/env python
# -*- coding: utf-8 -*-

from user_based_new import EnhancedCF

def train_specific_model(model_type="SVD", sim_method="Cosine"):
    """训练特定类型的模型"""
    print(f"开始训练 {model_type} 模型，使用 {sim_method} 相似度...")
    
    # 创建模型实例
    cf = EnhancedCF(
        n_sim_user=30,     # 相似用户数量
        n_rec_movie=10,    # 推荐电影数量
        pivot=0.8,         # 训练集比例
        n_factors=35,      # SVD因子数量
        sim_method=sim_method
    )
    
    # 加载数据集
    print("加载并拆分数据集...")
    cf.get_dataset()
    
    # 计算相似度
    print("计算用户相似度矩阵...")
    if model_type == "SVD":
        cf.calc_user_sim_svd()
    else:
        cf.calc_user_sim_sparse()
    
    # 保存模型
    model_name = f"{'svd' if model_type == 'SVD' else 'cf'}_{sim_method.lower()}"
    filepath = f"model/{model_name}_model.pkl"
    cf.save_model(filepath)
    print(f"模型已保存至: {filepath}")
    
    # 评估模型
    metrics = cf.evaluate_model()
    print("\n模型性能指标:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    return cf

if __name__ == "__main__":
    # 训练SVD模型，使用余弦相似度
    # train_specific_model("SVD", "Cosine")
    
    # 训练传统协同过滤模型，使用皮尔逊相关系数
    # train_specific_model("Traditional", "Pearson")
    
    # 训练所有可能的组合
    model_types = ["Traditional", "SVD"]
    sim_methods = ["Cosine", "Pearson", "Manhattan"]
    
    for model_type in model_types:
        for sim_method in sim_methods:
            print("\n" + "="*50)
            try:
                train_specific_model(model_type, sim_method)
            except Exception as e:
                print(f"训练失败: {model_type} 模型，使用 {sim_method} 相似度")
                print(f"错误: {e}")
            print("="*50)