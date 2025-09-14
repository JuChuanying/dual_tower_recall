#
# 训练入口
#
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import logging

from config.config import Config
from src.data_preprocessing import create_sample_data, preprocess_features, create_data_loaders
from src.model import DualTowerModel, ListwiseLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DualTowerTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        self.model = None
        self.optimizer = None
        self.criterion = None
        
    def prepare_data(self):
        """准备训练数据"""
        logger.info("准备数据...")
        
        # 创建示例数据
        interactions_df, user_df, item_df = create_sample_data()
        
        # 预处理特征
        user_df, item_df = preprocess_features(user_df, item_df)
        
        # 划分训练测试集
        train_data, test_data = train_test_split(
            interactions_df, test_size=0.2, random_state=42
        )
        
        # 计算物品流行度（用于热门物品打压）
        item_popularity = interactions_df.groupby('item_id').size()
        item_popularity = (item_popularity - item_popularity.min()) / (item_popularity.max() - item_popularity.min())
        self.item_popularity = item_popularity
        
        return train_data, test_data, user_df, item_df
    
    def setup_model(self, user_feature_dim, item_feature_dim):
        """设置模型"""
        logger.info("初始化模型...")
        
        self.model = DualTowerModel(
            user_feature_dim=user_feature_dim,
            item_feature_dim=item_feature_dim,
            embedding_dim=self.config.USER_EMBEDDING_DIM,
            hidden_dim=self.config.HIDDEN_DIM
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config.LEARNING_RATE,
            weight_decay=1e-5
        )
        
        self.criterion = ListwiseLoss()
        
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_labels = []
        all_scores = []
        
        for batch_idx, batch in enumerate(train_loader):
            # 正样本
            user_features = batch['user_features'].to(self.device)
            item_features = batch['item_features'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 前向传播
            user_embedding, item_embedding = self.model(user_features, item_features)
            positive_scores = self.model.compute_similarity(user_embedding, item_embedding)
            
            # 获取物品流行度
            item_ids = batch['item_id'].numpy()
            item_pop = torch.tensor([
                self.item_popularity.get(item_id, 0) for item_id in item_ids
            ], dtype=torch.float32).to(self.device)
            
            # 应用热门物品打压
            adjusted_scores = self.model.apply_popularity_penalty(positive_scores, item_pop)
            
            # 生成负样本（简单随机采样）
            negative_scores_list = []
            for _ in range(self.config.NEGATIVE_SAMPLE_SIZE):
                # 随机打乱物品特征作为负样本
                neg_item_features = item_features[torch.randperm(item_features.size(0))]
                _, neg_item_embedding = self.model(user_features, neg_item_features)
                neg_scores = self.model.compute_similarity(user_embedding, neg_item_embedding)
                
                # 对负样本也应用流行度打压
                neg_item_ids = batch['item_id'][torch.randperm(item_features.size(0))].numpy()
                neg_item_pop = torch.tensor([
                    self.item_popularity.get(item_id, 0) for item_id in neg_item_ids
                ], dtype=torch.float32).to(self.device)
                neg_adjusted_scores = self.model.apply_popularity_penalty(neg_scores, neg_item_pop)
                negative_scores_list.append(neg_adjusted_scores)
            
            negative_scores_tensor = torch.stack(negative_scores_list, dim=1)
            
            # 计算损失
            loss = self.criterion(adjusted_scores, negative_scores_tensor)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(adjusted_scores.detach().cpu().numpy())
            
            if batch_idx % 100 == 0:
                logger.info(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        auc_score = roc_auc_score(all_labels, all_scores)
        
        return avg_loss, auc_score
    
    def evaluate(self, test_loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_scores = []
        
        with torch.no_grad():
            for batch in test_loader:
                user_features = batch['user_features'].to(self.device)
                item_features = batch['item_features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                user_embedding, item_embedding = self.model(user_features, item_features)
                scores = self.model.compute_similarity(user_embedding, item_embedding)
                
                # 获取物品流行度并应用打压
                item_ids = batch['item_id'].numpy()
                item_pop = torch.tensor([
                    self.item_popularity.get(item_id, 0) for item_id in item_ids
                ], dtype=torch.float32).to(self.device)
                
                adjusted_scores = self.model.apply_popularity_penalty(scores, item_pop)
                
                # 简单损失计算用于评估
                loss = F.binary_cross_entropy_with_logits(adjusted_scores, labels)
                
                total_loss += loss.item()
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(adjusted_scores.cpu().numpy())
        
        avg_loss = total_loss / len(test_loader)
        auc_score = roc_auc_score(all_labels, all_scores)
        
        return avg_loss, auc_score
    
    def train(self):
        """完整训练流程"""
        # 准备数据
        train_data, test_data, user_df, item_df = self.prepare_data()
        
        # 设置模型
        self.setup_model(
            user_feature_dim=user_df.shape[1],
            item_feature_dim=item_df.shape[1]
        )
        
        # 创建数据加载器
        train_loader, test_loader = create_data_loaders(
            train_data, test_data, user_df, item_df, self.config
        )
        
        logger.info("开始训练...")
        best_auc = 0
        
        for epoch in range(self.config.EPOCHS):
            # 训练
            train_loss, train_auc = self.train_epoch(train_loader)
            
            # 评估
            test_loss, test_auc = self.evaluate(test_loader)
            
            logger.info(f'Epoch {epoch+1}/{self.config.EPOCHS}:')
            logger.info(f'  Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}')
            logger.info(f'  Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}')
            
            # 保存最佳模型
            if test_auc > best_auc:
                best_auc = test_auc
                torch.save(self.model.state_dict(), 'best_dual_tower_model.pth')
                logger.info(f'  Saved best model with AUC: {best_auc:.4f}')
        
        logger.info(f"训练完成，最佳AUC: {best_auc:.4f}")
        return best_auc

def main():
    config = Config()
    trainer = DualTowerTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
