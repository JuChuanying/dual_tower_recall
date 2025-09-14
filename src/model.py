# 
# 双塔网络
#
import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    """用户塔"""
    def __init__(self, feature_dim, embedding_dim, hidden_dim):
        super(UserTower, self).__init__()
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        
        # 用户特征编码器
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
    def forward(self, user_features):
        user_embedding = self.encoder(user_features)
        # L2归一化
        user_embedding = F.normalize(user_embedding, p=2, dim=1)
        return user_embedding

class ItemTower(nn.Module):
    """物品塔"""
    def __init__(self, feature_dim, embedding_dim, hidden_dim):
        super(ItemTower, self).__init__()
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        
        # 物品特征编码器
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
    def forward(self, item_features):
        item_embedding = self.encoder(item_features)
        # L2归一化
        item_embedding = F.normalize(item_embedding, p=2, dim=1)
        return item_embedding

class DualTowerModel(nn.Module):
    """双塔召回模型"""
    def __init__(self, user_feature_dim, item_feature_dim, embedding_dim, hidden_dim):
        super(DualTowerModel, self).__init__()
        self.user_tower = UserTower(user_feature_dim, embedding_dim, hidden_dim)
        self.item_tower = ItemTower(item_feature_dim, embedding_dim, hidden_dim)
        
        # 热门物品打压模块
        self.popularity_weight = nn.Parameter(torch.tensor(0.3))
        
    def forward(self, user_features, item_features):
        user_embedding = self.user_tower(user_features)
        item_embedding = self.item_tower(item_features)
        return user_embedding, item_embedding
    
    def get_user_embedding(self, user_features):
        """获取用户嵌入"""
        return self.user_tower(user_features)
    
    def get_item_embedding(self, item_features):
        """获取物品嵌入"""
        return self.item_tower(item_features)
    
    def compute_similarity(self, user_embedding, item_embedding):
        """计算相似度"""
        # 余弦相似度
        similarity = torch.sum(user_embedding * item_embedding, dim=1)
        return similarity
    
    def apply_popularity_penalty(self, similarity, item_popularity):
        """应用热门物品打压"""
        # 简单的热门物品打压：降低热门物品的相似度得分
        penalty = self.popularity_weight * item_popularity
        adjusted_similarity = similarity - penalty
        return adjusted_similarity

class ListwiseLoss(nn.Module):
    """Listwise损失函数"""
    def __init__(self, margin=0.1):
        super(ListwiseLoss, self).__init__()
        self.margin = margin
        
    def forward(self, positive_scores, negative_scores_list):
        """
        positive_scores: 正样本得分 [batch_size]
        negative_scores_list: 负样本得分列表 [batch_size, neg_samples]
        """
        batch_size = positive_scores.size(0)
        
        # 计算每个正样本与所有负样本的损失
        losses = []
        for i in range(batch_size):
            pos_score = positive_scores[i]
            neg_scores = negative_scores_list[i]
            
            # 对于每个负样本，计算hinge loss
            loss = torch.mean(F.relu(self.margin + neg_scores - pos_score))
            losses.append(loss)
            
        return torch.mean(torch.stack(losses))
