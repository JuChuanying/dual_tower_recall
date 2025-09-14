#
# 配置文件
#

import torch

class Config:
    # 模型参数
    USER_EMBEDDING_DIM = 128
    ITEM_EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    
    # 特征维度
    USER_FEATURE_DIM = 10  # 用户特征维度
    ITEM_FEATURE_DIM = 8   # 物品特征维度
    
    # 训练参数
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    EPOCHS = 50
    NEGATIVE_SAMPLE_SIZE = 5
    
    # 热门物品打压参数
    POPULARITY_THRESHOLD = 0.8
    POPULARITY_WEIGHT = 0.3
    
    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
