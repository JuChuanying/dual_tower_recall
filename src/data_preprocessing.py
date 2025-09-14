# 
# 数据处理
# 

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch

class RiceMallDataset(Dataset):
    def __init__(self, data, user_features, item_features, is_training=True):
        self.data = data
        self.user_features = user_features
        self.item_features = item_features
        self.is_training = is_training
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 用户特征
        user_id = row['user_id']
        user_feat = self.user_features.loc[user_id].values.astype(np.float32)
        
        # 物品特征
        item_id = row['item_id']
        item_feat = self.item_features.loc[item_id].values.astype(np.float32)
        
        if self.is_training:
            label = row['label']  # 1表示点击/购买，0表示未点击
            return {
                'user_id': user_id,
                'item_id': item_id,
                'user_features': torch.tensor(user_feat, dtype=torch.float32),
                'item_features': torch.tensor(item_feat, dtype=torch.float32),
                'label': torch.tensor(label, dtype=torch.float32)
            }
        else:
            return {
                'user_id': user_id,
                'item_id': item_id,
                'user_features': torch.tensor(user_feat, dtype=torch.float32),
                'item_features': torch.tensor(item_feat, dtype=torch.float32)
            }

def create_sample_data():
    """创建示例数据"""
    np.random.seed(42)
    
    # 用户特征：年龄、性别、消费水平、活跃度、历史购买次数等
    users = []
    for user_id in range(1000):
        user_data = {
            'user_id': user_id,
            'age': np.random.randint(18, 65),
            'gender': np.random.choice([0, 1]),
            'consumption_level': np.random.randint(1, 10),
            'activity_score': np.random.random(),
            'purchase_count': np.random.randint(0, 100),
            'avg_price': np.random.uniform(10, 500),
            'category_preference': np.random.random(),
            'time_on_site': np.random.randint(0, 3600),
            'device_type': np.random.choice([0, 1, 2])
        }
        users.append(user_data)
    
    # 物品特征：价格、类别、销量、评分、库存等
    items = []
    for item_id in range(500):
        item_data = {
            'item_id': item_id,
            'price': np.random.uniform(10, 1000),
            'category': np.random.choice(range(20)),
            'sales_count': np.random.randint(0, 1000),
            'rating': np.random.uniform(1, 5),
            'stock': np.random.randint(0, 100),
            'is_new': np.random.choice([0, 1]),
            'discount_rate': np.random.random(),
            'brand_popularity': np.random.random()
        }
        items.append(item_data)
    
    # 交互数据
    interactions = []
    user_df = pd.DataFrame(users)
    item_df = pd.DataFrame(items)
    
    # 模拟用户行为数据
    for _ in range(50000):
        user_id = np.random.randint(0, 1000)
        item_id = np.random.randint(0, 500)
        
        # 基于特征的简单点击概率
        user = user_df[user_df['user_id'] == user_id].iloc[0]
        item = item_df[item_df['item_id'] == item_id].iloc[0]
        
        # 点击概率基于用户偏好和物品特征
        click_prob = (
            0.1 + 
            0.2 * user['activity_score'] +
            0.1 * item['rating'] / 5 +
            0.1 * (1 - item['price'] / 1000) +
            0.1 * item['brand_popularity']
        )
        
        label = 1 if np.random.random() < click_prob else 0
        
        interactions.append({
            'user_id': user_id,
            'item_id': item_id,
            'label': label,
            'timestamp': np.random.randint(0, 1000000)
        })
    
    return pd.DataFrame(interactions), user_df, item_df

def preprocess_features(user_df, item_df):
    """预处理特征"""
    # 用户特征标准化
    user_scaler = StandardScaler()
    user_feature_cols = ['age', 'consumption_level', 'activity_score', 
                        'purchase_count', 'avg_price', 'category_preference', 
                        'time_on_site']
    user_df[user_feature_cols] = user_scaler.fit_transform(user_df[user_feature_cols])
    
    # 物品特征标准化
    item_scaler = StandardScaler()
    item_feature_cols = ['price', 'sales_count', 'rating', 'stock', 
                        'discount_rate', 'brand_popularity']
    item_df[item_feature_cols] = item_scaler.fit_transform(item_df[item_feature_cols])
    
    # 类别特征编码
    le = LabelEncoder()
    item_df['category_encoded'] = le.fit_transform(item_df['category'])
    
    # 设置索引
    user_df.set_index('user_id', inplace=True)
    item_df.set_index('item_id', inplace=True)
    
    return user_df, item_df

def create_data_loaders(train_data, test_data, user_features, item_features, config):
    """创建数据加载器"""
    train_dataset = RiceMallDataset(train_data, user_features, item_features, is_training=True)
    test_dataset = RiceMallDataset(test_data, user_features, item_features, is_training=True)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader
