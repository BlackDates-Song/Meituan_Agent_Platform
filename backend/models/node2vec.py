"""
图嵌入模型 - Node2Vec 风格的节点向量表示
用于生成用户、商家、菜品的向量表示
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import logging
import pickle
from typing import List, Dict, Tuple, Optional
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Node2VecModel:
    """Node2Vec 图嵌入模型"""
    
    def __init__(self, num_nodes: int, embedding_dim: int = 128):
        """
        初始化Node2Vec模型
        
        Args:
            num_nodes: 节点总数
            embedding_dim: 嵌入维度
        """
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.embeddings = None
        self.node_map = None  # 节点ID到索引的映射
        
    def random_walk(self, graph_dict: Dict[int, List[int]],
                   start_node: int,
                   walk_length: int = 10,
                   p: float = 1.0,
                   q: float = 1.0) -> List[int]:
        """
        随机游走
        
        Args:
            graph_dict: 图的邻接表 {node_id: [neighbor_ids]}
            start_node: 起始节点
            walk_length: 游走长度
            p: 返回参数
            q: 进出参数
            
        Returns:
            游走路径
        """
        walk = [start_node]
        
        while len(walk) < walk_length:
            curr = walk[-1]
            neighbors = graph_dict.get(curr, [])
            
            if len(neighbors) == 0:
                break
                
            if len(walk) == 1:
                # 第一步，均匀随机选择
                next_node = random.choice(neighbors)
            else:
                # 根据p和q参数选择下一个节点
                prev = walk[-2]
                
                # 计算每个邻居的权重
                weights = []
                for neighbor in neighbors:
                    if neighbor == prev:
                        weight = 1.0 / p  # 返回上一个节点
                    elif neighbor in graph_dict.get(prev, []):
                        weight = 1.0  # 邻居的邻居
                    else:
                        weight = 1.0 / q  # 更远的节点
                    weights.append(weight)
                    
                # 归一化权重并选择
                weights = np.array(weights)
                weights = weights / weights.sum()
                next_node = np.random.choice(neighbors, p=weights)
                
            walk.append(next_node)
            
        return walk
        
    def generate_walks(self, graph_dict: Dict[int, List[int]],
                      num_walks: int = 10,
                      walk_length: int = 10,
                      p: float = 1.0,
                      q: float = 1.0) -> List[List[int]]:
        """
        为所有节点生成随机游走
        
        Args:
            graph_dict: 图的邻接表
            num_walks: 每个节点的游走次数
            walk_length: 每次游走的长度
            p: 返回参数
            q: 进出参数
            
        Returns:
            所有游走路径列表
        """
        logger.info(f"生成随机游走 (节点数={len(graph_dict)}, 每节点{num_walks}次, 长度={walk_length})...")
        
        walks = []
        nodes = list(graph_dict.keys())
        
        for _ in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = self.random_walk(graph_dict, node, walk_length, p, q)
                walks.append(walk)
                
        logger.info(f"✓ 生成了 {len(walks)} 条游走路径")
        return walks
        
    def train_skip_gram(self, walks: List[List[int]],
                       window_size: int = 5,
                       num_epochs: int = 5,
                       learning_rate: float = 0.01,
                       negative_samples: int = 5):
        """
        使用Skip-Gram训练嵌入
        
        Args:
            walks: 游走路径列表
            window_size: 上下文窗口大小
            num_epochs: 训练轮数
            learning_rate: 学习率
            negative_samples: 负采样数量
        """
        logger.info("训练Skip-Gram模型...")
        
        # 初始化嵌入矩阵
        self.embeddings = np.random.randn(self.num_nodes, self.embedding_dim) * 0.01
        
        # 准备训练数据
        training_pairs = []
        for walk in walks:
            for i, center in enumerate(walk):
                # 获取上下文窗口
                start = max(0, i - window_size)
                end = min(len(walk), i + window_size + 1)
                
                for j in range(start, end):
                    if i != j:
                        context = walk[j]
                        training_pairs.append((center, context, 1))  # 正样本
                        
        logger.info(f"训练样本数: {len(training_pairs)}")
        
        # 简化的SGD训练
        for epoch in range(num_epochs):
            random.shuffle(training_pairs)
            total_loss = 0
            
            for center, context, label in training_pairs[:10000]:  # 限制样本数以加快速度
                # 前向传播
                center_emb = self.embeddings[center]
                context_emb = self.embeddings[context]
                
                # 计算相似度
                similarity = np.dot(center_emb, context_emb)
                prediction = 1 / (1 + np.exp(-similarity))
                
                # 损失
                loss = -label * np.log(prediction + 1e-10) - (1 - label) * np.log(1 - prediction + 1e-10)
                total_loss += loss
                
                # 反向传播
                gradient = (prediction - label)
                center_grad = gradient * context_emb
                context_grad = gradient * center_emb
                
                # 更新
                self.embeddings[center] -= learning_rate * center_grad
                self.embeddings[context] -= learning_rate * context_grad
                
            avg_loss = total_loss / min(10000, len(training_pairs))
            logger.info(f"Epoch {epoch+1}/{num_epochs}, 平均损失: {avg_loss:.4f}")
            
        logger.info("✓ 训练完成")
        
    def get_embedding(self, node_id: int) -> np.ndarray:
        """
        获取节点的嵌入向量
        
        Args:
            node_id: 节点ID
            
        Returns:
            嵌入向量
        """
        if self.embeddings is None:
            raise RuntimeError("模型尚未训练")
            
        return self.embeddings[node_id]
        
    def get_similar_nodes(self, node_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        找到最相似的节点
        
        Args:
            node_id: 节点ID
            top_k: 返回前k个相似节点
            
        Returns:
            [(节点ID, 相似度), ...]
        """
        if self.embeddings is None:
            raise RuntimeError("模型尚未训练")
            
        node_emb = self.embeddings[node_id]
        
        # 计算余弦相似度
        similarities = np.dot(self.embeddings, node_emb)
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(node_emb)
        similarities = similarities / (norms + 1e-10)
        
        # 排序
        top_indices = np.argsort(similarities)[::-1][1:top_k+1]  # 排除自己
        
        results = [(idx, similarities[idx]) for idx in top_indices]
        return results
        
    def save_embeddings(self, output_path: str):
        """保存嵌入向量"""
        if self.embeddings is None:
            raise RuntimeError("模型尚未训练")
            
        np.save(output_path, self.embeddings)
        logger.info(f"嵌入向量已保存到: {output_path}")
        
    def load_embeddings(self, input_path: str):
        """加载嵌入向量"""
        self.embeddings = np.load(input_path)
        logger.info(f"嵌入向量已从 {input_path} 加载")


def build_graph_from_db(db_path: str, graph_type: str = 'user_poi') -> Dict[int, List[int]]:
    """
    从数据库构建图
    
    Args:
        db_path: 数据库路径
        graph_type: 图类型 ('user_poi', 'user_spu', 'poi_spu')
        
    Returns:
        图的邻接表
    """
    import sqlite3
    
    logger.info(f"构建 {graph_type} 图...")
    
    conn = sqlite3.connect(db_path)
    
    if graph_type == 'user_poi':
        # 用户-商家二部图
        query = "SELECT user_id, wm_poi_id FROM orders_train"
    elif graph_type == 'user_spu':
        # 用户-菜品二部图
        query = """
            SELECT o.user_id, os.wm_food_spu_id
            FROM orders_spu_train os
            JOIN orders_train o ON os.wm_order_id = o.wm_order_id
        """
    else:
        raise ValueError(f"不支持的图类型: {graph_type}")
        
    import pandas as pd
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # 构建邻接表
    graph = {}
    
    for _, row in df.iterrows():
        src, dst = row.iloc[0], row.iloc[1]
        
        # 双向边（二部图）
        if src not in graph:
            graph[src] = []
        if dst not in graph:
            graph[dst] = []
            
        if dst not in graph[src]:
            graph[src].append(dst)
        if src not in graph[dst]:
            graph[dst].append(src)
            
    logger.info(f"✓ 图构建完成: {len(graph)} 个节点")
    return graph


def train_node2vec_model(db_path: str, output_dir: Path, graph_type: str = 'user_poi', 
                        sample_nodes: int = 10000):
    """
    训练Node2Vec模型
    
    Args:
        db_path: 数据库路径
        output_dir: 输出目录
        graph_type: 图类型
        sample_nodes: 采样节点数量（避免内存和时间溢出）
    """
    logger.info("=" * 60)
    logger.info(f"训练 {graph_type} Node2Vec 模型")
    logger.info("=" * 60)
    
    # 构建图
    graph = build_graph_from_db(db_path, graph_type)
    
    # 如果节点数过多，进行采样
    all_nodes = list(graph.keys())
    if len(all_nodes) > sample_nodes:
        logger.info(f"节点数 {len(all_nodes)} 超过采样限制 {sample_nodes}，进行采样...")
        import random
        random.seed(42)
        sampled_nodes = set(random.sample(all_nodes, sample_nodes))
        
        # 构建采样子图
        sampled_graph = {}
        for node in sampled_nodes:
            if node in graph:
                # 只保留采样节点的邻居
                sampled_neighbors = [n for n in graph[node] if n in sampled_nodes]
                if sampled_neighbors:
                    sampled_graph[node] = sampled_neighbors
        
        graph = sampled_graph
        logger.info(f"采样后图节点数: {len(graph)}")
    
    # 创建模型
    num_nodes = max(max(graph.keys()), max(max(neighbors) for neighbors in graph.values() if neighbors)) + 1
    model = Node2VecModel(num_nodes=num_nodes, embedding_dim=128)
    
    # 生成随机游走（减少游走次数和长度）
    walks = model.generate_walks(graph, num_walks=5, walk_length=8, p=1.0, q=1.0)
    
    # 训练模型（减少轮数）
    model.train_skip_gram(walks, window_size=5, num_epochs=2, learning_rate=0.01)
    
    # 保存嵌入
    output_path = output_dir / f"{graph_type}_node2vec_embeddings.npy"
    model.save_embeddings(str(output_path))
    
    logger.info(f"✓ {graph_type} 模型训练完成")
    
    return model


def main():
    """主函数"""
    # 项目根目录
    project_root = Path(__file__).parent.parent.parent
    db_path = project_root / "data" / "db" / "meituan.db"
    output_dir = project_root / "backend" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not db_path.exists():
        logger.error(f"数据库不存在: {db_path}")
        logger.error("请先运行 load_data.py 加载数据")
        return
        
    # 训练用户-商家图嵌入
    poi_model = train_node2vec_model(str(db_path), output_dir, 'user_poi')
    
    # 训练用户-菜品图嵌入
    spu_model = train_node2vec_model(str(db_path), output_dir, 'user_spu')
    
    logger.info("\n" + "=" * 60)
    logger.info("所有图嵌入模型训练完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
