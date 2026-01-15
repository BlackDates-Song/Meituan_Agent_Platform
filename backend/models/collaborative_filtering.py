"""
协同过滤推荐模型
实现基于用户和基于物品的协同过滤算法
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import sqlite3
import pickle
import logging
from typing import List, Tuple, Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CollaborativeFiltering:
    """协同过滤基类"""
    
    def __init__(self, db_path: str):
        """
        初始化协同过滤模型
        
        Args:
            db_path: SQLite 数据库路径
        """
        self.db_path = Path(db_path)
        self.similarity_matrix = None
        self.user_item_matrix = None
        self.user_ids = None
        self.item_ids = None
        
    def load_data(self, table_name: str, user_col: str, item_col: str):
        """
        从数据库加载交互数据
        
        Args:
            table_name: 表名
            user_col: 用户列名
            item_col: 物品列名
        """
        conn = sqlite3.connect(self.db_path)
        
        query = f"""
            SELECT {user_col}, {item_col}, COUNT(*) as count
            FROM {table_name}
            GROUP BY {user_col}, {item_col}
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        logger.info(f"加载交互数据: {len(df)} 条记录")
        
        # 创建用户-物品矩阵
        self.user_ids = df[user_col].unique()
        self.item_ids = df[item_col].unique()
        
        user_map = {uid: idx for idx, uid in enumerate(self.user_ids)}
        item_map = {iid: idx for idx, iid in enumerate(self.item_ids)}
        
        row_indices = df[user_col].map(user_map)
        col_indices = df[item_col].map(item_map)
        data = df['count'].values
        
        self.user_item_matrix = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(len(self.user_ids), len(self.item_ids))
        )
        
        logger.info(f"用户-物品矩阵: {self.user_item_matrix.shape}")
        
    def compute_similarity(self):
        """计算相似度矩阵（需要在子类中实现）"""
        raise NotImplementedError
        
    def recommend(self, user_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """推荐物品（需要在子类中实现）"""
        raise NotImplementedError
        
    def save_model(self, model_path: str):
        """保存模型"""
        model_data = {
            'similarity_matrix': self.similarity_matrix,
            'user_item_matrix': self.user_item_matrix,
            'user_ids': self.user_ids,
            'item_ids': self.item_ids
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"模型已保存到: {model_path}")
        
    def load_model(self, model_path: str):
        """加载模型"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.similarity_matrix = model_data['similarity_matrix']
        self.user_item_matrix = model_data['user_item_matrix']
        self.user_ids = model_data['user_ids']
        self.item_ids = model_data['item_ids']
        
        logger.info(f"模型已从 {model_path} 加载")


class UserBasedCF(CollaborativeFiltering):
    """基于用户的协同过滤"""
    
    def __init__(self, db_path: str, sample_size: int = 10000):
        """
        初始化
        
        Args:
            db_path: 数据库路径
            sample_size: 用户采样数量（避免内存溢出）
        """
        super().__init__(db_path)
        self.sample_size = sample_size
    
    def compute_similarity(self, metric: str = 'cosine'):
        """
        计算用户之间的相似度
        
        Args:
            metric: 相似度度量方法 ('cosine' 或 'pearson')
        """
        logger.info(f"计算用户相似度 (方法: {metric})...")
        
        # 如果用户数量过多，进行采样
        n_users = self.user_item_matrix.shape[0]
        if n_users > self.sample_size:
            logger.info(f"用户数量 {n_users} 超过采样限制 {self.sample_size}，进行随机采样...")
            import random
            sample_indices = random.sample(range(n_users), self.sample_size)
            sampled_matrix = self.user_item_matrix[sample_indices, :]
            sampled_user_ids = self.user_ids[sample_indices]
        else:
            sampled_matrix = self.user_item_matrix
            sampled_user_ids = self.user_ids
        
        if metric == 'cosine':
            self.similarity_matrix = cosine_similarity(sampled_matrix)
        if metric == 'cosine':
            self.similarity_matrix = cosine_similarity(sampled_matrix)
        elif metric == 'pearson':
            # Pearson 相关系数
            user_mean = np.array(sampled_matrix.mean(axis=1)).flatten()
            user_item_centered = sampled_matrix.toarray() - user_mean[:, np.newaxis]
            self.similarity_matrix = np.corrcoef(user_item_centered)
        else:
            raise ValueError(f"不支持的相似度度量: {metric}")
            
        # 将对角线设为0（不考虑自己和自己的相似度）
        np.fill_diagonal(self.similarity_matrix, 0)
        
        # 更新用户ID列表为采样后的
        self.user_ids = sampled_user_ids
        self.user_item_matrix = sampled_matrix
        
        logger.info(f"✓ 相似度矩阵计算完成: {self.similarity_matrix.shape}")
        
    def recommend(self, user_id: int, top_k: int = 10,
                  exclude_interacted: bool = True) -> List[Tuple[int, float]]:
        """
        为用户推荐物品
        
        Args:
            user_id: 用户ID
            top_k: 返回Top K个推荐
            exclude_interacted: 是否排除已交互物品
            
        Returns:
            [(物品ID, 预测评分), ...]
        """
        # 找到用户索引
        try:
            user_idx = np.where(self.user_ids == user_id)[0][0]
        except IndexError:
            logger.warning(f"用户 {user_id} 不存在")
            return []
            
        # 找到最相似的用户
        user_similarities = self.similarity_matrix[user_idx]
        
        # 获取用户的交互历史
        user_interactions = self.user_item_matrix[user_idx].toarray().flatten()
        
        # 计算预测评分
        # 评分 = Σ(相似度 * 用户对物品的评分) / Σ相似度
        similar_users_interactions = self.user_item_matrix.toarray()
        
        # 加权求和
        weighted_sum = np.dot(user_similarities, similar_users_interactions)
        similarity_sum = np.abs(user_similarities).sum()
        
        if similarity_sum > 0:
            predicted_scores = weighted_sum / similarity_sum
        else:
            predicted_scores = np.zeros(len(self.item_ids))
            
        # 如果排除已交互物品
        if exclude_interacted:
            predicted_scores[user_interactions > 0] = -np.inf
            
        # 获取Top K
        top_indices = np.argsort(predicted_scores)[::-1][:top_k]
        recommendations = [
            (self.item_ids[idx], predicted_scores[idx])
            for idx in top_indices
            if predicted_scores[idx] > 0
        ]
        
        return recommendations


class ItemBasedCF(CollaborativeFiltering):
    """基于物品的协同过滤"""
    
    def __init__(self, db_path: str, sample_size: int = 20000):
        """
        初始化
        
        Args:
            db_path: 数据库路径
            sample_size: 物品采样数量（避免内存溢出）
        """
        super().__init__(db_path)
        self.sample_size = sample_size
    
    def compute_similarity(self, metric: str = 'cosine'):
        """
        计算物品之间的相似度
        
        Args:
            metric: 相似度度量方法
        """
        logger.info(f"计算物品相似度 (方法: {metric})...")
        
        # 转置矩阵，以物品为行
        item_user_matrix = self.user_item_matrix.T
        
        # 如果物品数量过多，进行采样
        n_items = item_user_matrix.shape[0]
        if n_items > self.sample_size:
            logger.info(f"物品数量 {n_items} 超过采样限制 {self.sample_size}，进行随机采样...")
            import random
            sample_indices = random.sample(range(n_items), self.sample_size)
            sampled_matrix = item_user_matrix[sample_indices, :]
            sampled_item_ids = self.item_ids[sample_indices]
        else:
            sampled_matrix = item_user_matrix
            sampled_item_ids = self.item_ids
        
        if metric == 'cosine':
            self.similarity_matrix = cosine_similarity(sampled_matrix)
        if metric == 'cosine':
            self.similarity_matrix = cosine_similarity(sampled_matrix)
        elif metric == 'pearson':
            item_mean = np.array(sampled_matrix.mean(axis=1)).flatten()
            item_user_centered = sampled_matrix.toarray() - item_mean[:, np.newaxis]
            self.similarity_matrix = np.corrcoef(item_user_centered)
        else:
            raise ValueError(f"不支持的相似度度量: {metric}")
            
        np.fill_diagonal(self.similarity_matrix, 0)
        
        # 更新物品ID列表为采样后的
        self.item_ids = sampled_item_ids
        
        logger.info(f"✓ 相似度矩阵计算完成: {self.similarity_matrix.shape}")
        
    def recommend(self, user_id: int, top_k: int = 10,
                  exclude_interacted: bool = True) -> List[Tuple[int, float]]:
        """
        为用户推荐物品
        
        Args:
            user_id: 用户ID
            top_k: 返回Top K个推荐
            exclude_interacted: 是否排除已交互物品
            
        Returns:
            [(物品ID, 预测评分), ...]
        """
        try:
            user_idx = np.where(self.user_ids == user_id)[0][0]
        except IndexError:
            logger.warning(f"用户 {user_id} 不存在")
            return []
            
        # 获取用户的交互历史
        user_interactions = self.user_item_matrix[user_idx].toarray().flatten()
        interacted_items = np.where(user_interactions > 0)[0]
        
        # 对于用户交互过的每个物品，找到相似物品
        predicted_scores = np.zeros(len(self.item_ids))
        
        for item_idx in interacted_items:
            # 该物品与所有物品的相似度
            item_similarities = self.similarity_matrix[item_idx]
            
            # 加权累加
            predicted_scores += item_similarities * user_interactions[item_idx]
            
        # 归一化
        interacted_count = len(interacted_items)
        if interacted_count > 0:
            predicted_scores /= interacted_count
            
        # 排除已交互物品
        if exclude_interacted:
            predicted_scores[user_interactions > 0] = -np.inf
            
        # 获取Top K
        top_indices = np.argsort(predicted_scores)[::-1][:top_k]
        recommendations = [
            (self.item_ids[idx], predicted_scores[idx])
            for idx in top_indices
            if predicted_scores[idx] > 0
        ]
        
        return recommendations


class HybridCF:
    """混合协同过滤（结合用户CF和物品CF）"""
    
    def __init__(self, user_cf: UserBasedCF, item_cf: ItemBasedCF,
                 user_weight: float = 0.5):
        """
        初始化混合模型
        
        Args:
            user_cf: 基于用户的CF模型
            item_cf: 基于物品的CF模型
            user_weight: 用户CF的权重（0-1之间）
        """
        self.user_cf = user_cf
        self.item_cf = item_cf
        self.user_weight = user_weight
        self.item_weight = 1 - user_weight
        
    def recommend(self, user_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        混合推荐
        
        Args:
            user_id: 用户ID
            top_k: 返回Top K个推荐
            
        Returns:
            [(物品ID, 预测评分), ...]
        """
        # 获取两个模型的推荐
        user_recs = self.user_cf.recommend(user_id, top_k * 2)
        item_recs = self.item_cf.recommend(user_id, top_k * 2)
        
        # 合并评分
        score_dict = {}
        
        for item_id, score in user_recs:
            score_dict[item_id] = score * self.user_weight
            
        for item_id, score in item_recs:
            if item_id in score_dict:
                score_dict[item_id] += score * self.item_weight
            else:
                score_dict[item_id] = score * self.item_weight
                
        # 排序并返回Top K
        sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]


def train_poi_recommender(db_path: str, model_dir: Path):
    """训练商家推荐模型"""
    logger.info("=" * 60)
    logger.info("训练商家推荐模型 (POI)")
    logger.info("=" * 60)
    
    # 基于用户的CF
    logger.info("\n1. 训练基于用户的协同过滤...")
    user_cf = UserBasedCF(db_path)
    user_cf.load_data('orders_train', 'user_id', 'wm_poi_id')
    user_cf.compute_similarity('cosine')
    user_cf.save_model(model_dir / 'user_based_poi_cf.pkl')
    
    # 基于物品的CF
    logger.info("\n2. 训练基于物品的协同过滤...")
    item_cf = ItemBasedCF(db_path)
    item_cf.load_data('orders_train', 'user_id', 'wm_poi_id')
    item_cf.compute_similarity('cosine')
    item_cf.save_model(model_dir / 'item_based_poi_cf.pkl')
    
    logger.info("\n✓ 商家推荐模型训练完成")
    
    return user_cf, item_cf


def train_spu_recommender(db_path: str, model_dir: Path):
    """训练菜品推荐模型"""
    logger.info("=" * 60)
    logger.info("训练菜品推荐模型 (SPU)")
    logger.info("=" * 60)
    
    # 需要先关联订单表获取 user_id
    conn = sqlite3.connect(db_path)
    
    # 创建临时表
    conn.execute("""
        CREATE TEMP TABLE user_spu_interactions AS
        SELECT o.user_id, os.wm_food_spu_id
        FROM orders_spu_train os
        JOIN orders_train o ON os.wm_order_id = o.wm_order_id
    """)
    
    conn.close()
    
    # 基于用户的CF
    logger.info("\n1. 训练基于用户的协同过滤...")
    user_cf = UserBasedCF(db_path)
    user_cf.load_data('user_spu_interactions', 'user_id', 'wm_food_spu_id')
    user_cf.compute_similarity('cosine')
    user_cf.save_model(model_dir / 'user_based_spu_cf.pkl')
    
    # 基于物品的CF
    logger.info("\n2. 训练基于物品的协同过滤...")
    item_cf = ItemBasedCF(db_path)
    item_cf.load_data('user_spu_interactions', 'user_id', 'wm_food_spu_id')
    item_cf.compute_similarity('cosine')
    item_cf.save_model(model_dir / 'item_based_spu_cf.pkl')
    
    logger.info("\n✓ 菜品推荐模型训练完成")
    
    return user_cf, item_cf


def main():
    """主函数"""
    # 项目根目录
    project_root = Path(__file__).parent.parent.parent
    db_path = project_root / "data" / "db" / "meituan.db"
    model_dir = project_root / "backend" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    if not db_path.exists():
        logger.error(f"数据库不存在: {db_path}")
        logger.error("请先运行 load_data.py 加载数据")
        return
        
    # 训练商家推荐模型
    poi_user_cf, poi_item_cf = train_poi_recommender(str(db_path), model_dir)
    
    # 训练菜品推荐模型
    spu_user_cf, spu_item_cf = train_spu_recommender(str(db_path), model_dir)
    
    # 测试推荐
    logger.info("\n" + "=" * 60)
    logger.info("推荐示例")
    logger.info("=" * 60)
    
    # 随机选择一个用户
    test_user_id = poi_user_cf.user_ids[0]
    logger.info(f"\n为用户 {test_user_id} 推荐商家:")
    
    poi_recs = poi_user_cf.recommend(test_user_id, top_k=5)
    for rank, (poi_id, score) in enumerate(poi_recs, 1):
        logger.info(f"  {rank}. POI {poi_id}: {score:.4f}")
        
    logger.info(f"\n为用户 {test_user_id} 推荐菜品:")
    spu_recs = spu_user_cf.recommend(test_user_id, top_k=5)
    for rank, (spu_id, score) in enumerate(spu_recs, 1):
        logger.info(f"  {rank}. SPU {spu_id}: {score:.4f}")
        
    logger.info("\n" + "=" * 60)
    logger.info("所有模型训练完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
