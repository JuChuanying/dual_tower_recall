#
# 推理脚本
#
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from config.config import Config
from src.model import DualTowerModel

class DualTowerInference:
    def __init__(self, model_path, user_features, item_features, config):
        self.config = config
        self.device = config.DEVICE
        
        # 加载模型
        self.model = DualTowerModel(
            user_feature_dim=user_features.shape[1],
            item_feature_dim=item_features.shape[1],
            embedding_dim=config.USER_EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.user_features = user_features
        self.item_features = item_features
        
        # 预计算所有物品嵌入
        self.item_embeddings = self._precompute_item_embeddings()
        
    def _precompute_item_embeddings(self):
        """预计算所有物品嵌入"""
        print("预计算物品嵌入...")
        item_embeddings = {}
        
        with torch.no_grad():
            for item_id in self.item_features.index:
                item_feat = torch.tensor(
                    self.item_features.loc[item_id].values, 
                    dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                
                item_embedding = self.model.get_item_embedding(item_feat)
                item_embeddings[item_id] = item_embedding.cpu().numpy().flatten()
        
        return item_embeddings
    
    def get_user_embedding(self, user_id):
        """获取用户嵌入"""
        if user_id not in self.user_features.index:
            raise ValueError(f"User {user_id} not found in features")
            
        user_feat = torch.tensor(
            self.user_features.loc[user_id].values, 
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            user_embedding = self.model.get_user_embedding(user_feat)
            
        return user_embedding.cpu().numpy().flatten()
    
    def recall_items(self, user_id, top_k=50, exclude_items=None):
        """
        为用户召回物品
        
        Args:
            user_id: 用户ID
            top_k: 返回top K个物品
            exclude_items: 需要排除的物品列表
            
        Returns:
            list: [(item_id, score), ...]
        """
        # 获取用户嵌入
        user_embedding = self.get_user_embedding(user_id)
        
        # 计算与所有物品的相似度
        similarities = {}
        for item_id, item_embedding in self.item_embeddings.items():
            if exclude_items and item_id in exclude_items:
                continue
                
            # 余弦相似度
            similarity = np.dot(user_embedding, item_embedding) / (
                np.linalg.norm(user_embedding) * np.linalg.norm(item_embedding)
            )
            similarities[item_id] = similarity
        
        # 排序并返回top K
        sorted_items = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]
    
    def batch_recall(self, user_ids, top_k=50):
        """批量召回"""
        results = {}
        for user_id in user_ids:
            results[user_id] = self.recall_items(user_id, top_k)
        return results

# 使用示例
def demo_inference():
    from src.data_preprocessing import create_sample_data, preprocess_features
    
    # 加载数据
    _, user_df, item_df = create_sample_data()
    user_df, item_df = preprocess_features(user_df, item_df)
    
    # 初始化推理器
    config = Config()
    inference = DualTowerInference(
        model_path='best_dual_tower_model.pth',
        user_features=user_df,
        item_features=item_df,
        config=config
    )
    
    # 为用户召回物品
    user_id = 0
    recommendations = inference.recall_items(user_id, top_k=10)
    
    print(f"为用户 {user_id} 推荐的物品:")
    for item_id, score in recommendations:
        print(f"  物品 {item_id}: 得分 {score:.4f}")

if __name__ == "__main__":
    demo_inference()
