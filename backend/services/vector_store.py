"""
向量数据库服务 - 基于 ChromaDB
支持商家、菜品、评论的向量化存储和语义检索
"""

import os
import sqlite3
from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("Warning: ChromaDB not installed. Vector search will be disabled.")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class VectorStore:
    """向量数据库封装类"""
    
    def __init__(self, persist_directory: str = "data/vector_db"):
        """
        初始化向量数据库
        
        Args:
            persist_directory: 向量数据库持久化目录
        """
        self.persist_directory = persist_directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        if not CHROMA_AVAILABLE:
            self.client = None
            return
            
        # 初始化ChromaDB客户端
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # 创建集合
        self.poi_collection = self.client.get_or_create_collection(
            name="pois",
            metadata={"description": "商家向量索引"}
        )
        
        self.spu_collection = self.client.get_or_create_collection(
            name="spus",
            metadata={"description": "菜品向量索引"}
        )
        
        self.comment_collection = self.client.get_or_create_collection(
            name="comments",
            metadata={"description": "评论向量索引（模拟数据）"}
        )
        
        # OpenAI客户端（用于生成embeddings）
        self.openai_client = None
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        if api_key and OPENAI_AVAILABLE:
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
            self.openai_client = OpenAI(api_key=api_key, base_url=base_url)
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        生成文本的向量表示
        
        Args:
            text: 输入文本
            
        Returns:
            向量列表，如果失败返回None
        """
        if not self.openai_client:
            # 简单的TF-IDF风格的embedding（fallback）
            words = set(text.lower().split())
            # 使用简单的hash来生成固定维度的向量
            embedding = [0.0] * 384
            for word in words:
                idx = hash(word) % 384
                embedding[idx] += 1.0
            # 归一化
            norm = sum(x**2 for x in embedding) ** 0.5
            if norm > 0:
                embedding = [x / norm for x in embedding]
            return embedding
        
        try:
            # 使用OpenAI兼容API生成embedding
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding generation failed: {e}")
            # Fallback到简单方法
            words = set(text.lower().split())
            embedding = [0.0] * 384
            for word in words:
                idx = hash(word) % 384
                embedding[idx] += 1.0
            norm = sum(x**2 for x in embedding) ** 0.5
            if norm > 0:
                embedding = [x / norm for x in embedding]
            return embedding
    
    def index_pois(self, db_path: str = "data/db/meituan.db"):
        """
        索引所有商家数据
        
        Args:
            db_path: 数据库路径
        """
        if not self.client:
            print("ChromaDB not available")
            return
            
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 查询商家数据
        cursor.execute("""
            SELECT wm_poi_id, wm_poi_name, primary_first_tag_name, 
                   primary_second_tag_name, primary_third_tag_name, 
                   poi_score, 0 as price, aor_id as district
            FROM pois
        """)
        
        pois = cursor.fetchall()
        print(f"Indexing {len(pois)} POIs...")
        
        batch_size = 100
        for i in range(0, len(pois), batch_size):
            batch = pois[i:i+batch_size]
            
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            for poi in batch:
                poi_id, name, cate1, cate2, cate3, score, price, district = poi
                
                # 构建文本描述
                text = f"{name} {cate1 or ''} {cate2 or ''} {cate3 or ''} {district or ''}"
                
                # 生成embedding
                embedding = self._generate_embedding(text)
                if embedding is None:
                    continue
                
                ids.append(f"poi_{poi_id}")
                embeddings.append(embedding)
                documents.append(text)
                metadatas.append({
                    "poi_id": str(poi_id),
                    "name": name or "",
                    "category": f"{cate1 or ''}/{cate2 or ''}/{cate3 or ''}",
                    "score": float(score) if score else 0.0,
                    "price": float(price) if price else 0.0,
                    "district": district or ""
                })
            
            if ids:
                self.poi_collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
        
        conn.close()
        print(f"POI indexing completed: {len(pois)} items")
    
    def index_spus(self, db_path: str = "data/db/meituan.db"):
        """
        索引所有菜品数据
        
        Args:
            db_path: 数据库路径
        """
        if not self.client:
            print("ChromaDB not available")
            return
            
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 查询菜品数据
        cursor.execute("""
            SELECT wm_food_spu_id, wm_food_spu_name, category, ingredients, 
                   taste, stand_food_id, stand_food_name
            FROM spus
        """)
        
        spus = cursor.fetchall()
        print(f"Indexing {len(spus)} SPUs...")
        
        batch_size = 100
        for i in range(0, len(spus), batch_size):
            batch = spus[i:i+batch_size]
            
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            for spu in batch:
                (spu_id, spu_name, category, ingredients, taste,
                 stand_food_id, stand_food_name) = spu
                
                # 构建文本描述
                text = f"{spu_name or ''} {category or ''} {ingredients or ''} {taste or ''} {stand_food_name or ''}"
                
                # 生成embedding
                embedding = self._generate_embedding(text)
                if embedding is None:
                    continue
                
                ids.append(f"spu_{spu_id}")
                embeddings.append(embedding)
                documents.append(text)
                metadatas.append({
                    "spu_id": str(spu_id),
                    "name": spu_name or "",
                    "category": category or "",
                    "ingredients": ingredients or "",
                    "taste": taste or ""
                })
            
            if ids:
                self.spu_collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
        
        conn.close()
        print(f"SPU indexing completed: {len(spus)} items")
    
    def search_pois(self, query: str, n_results: int = 10, 
                    filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        语义搜索商家
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            filters: 元数据过滤条件
            
        Returns:
            商家列表
        """
        if not self.client:
            return []
        
        query_embedding = self._generate_embedding(query)
        if query_embedding is None:
            return []
        
        results = self.poi_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filters
        )
        
        pois = []
        if results['ids'] and len(results['ids']) > 0:
            for i, poi_id in enumerate(results['ids'][0]):
                pois.append({
                    "poi_id": results['metadatas'][0][i]['poi_id'],
                    "name": results['metadatas'][0][i]['name'],
                    "category": results['metadatas'][0][i]['category'],
                    "score": results['metadatas'][0][i]['score'],
                    "price": results['metadatas'][0][i]['price'],
                    "district": results['metadatas'][0][i]['district'],
                    "similarity": 1.0 - results['distances'][0][i] if 'distances' in results else 1.0
                })
        
        return pois
    
    def search_spus(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        语义搜索菜品
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            
        Returns:
            菜品列表
        """
        if not self.client:
            return []
        
        query_embedding = self._generate_embedding(query)
        if query_embedding is None:
            return []
        
        results = self.spu_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        spus = []
        if results['ids'] and len(results['ids']) > 0:
            for i, spu_id in enumerate(results['ids'][0]):
                spus.append({
                    "spu_id": results['metadatas'][0][i]['spu_id'],
                    "name": results['metadatas'][0][i]['name'],
                    "tag": results['metadatas'][0][i]['tag'],
                    "category": results['metadatas'][0][i]['category'],
                    "similarity": 1.0 - results['distances'][0][i] if 'distances' in results else 1.0
                })
        
        return spus
    
    def get_stats(self) -> Dict[str, int]:
        """获取向量库统计信息"""
        if not self.client:
            return {"pois": 0, "spus": 0, "comments": 0}
        
        return {
            "pois": self.poi_collection.count(),
            "spus": self.spu_collection.count(),
            "comments": self.comment_collection.count()
        }


def build_vector_index():
    """构建向量索引的主函数"""
    print("Building vector index...")
    
    vector_store = VectorStore()
    
    if not vector_store.client:
        print("ChromaDB not available. Please install: pip install chromadb")
        return
    
    # 索引商家和菜品
    vector_store.index_pois()
    vector_store.index_spus()
    
    # 显示统计
    stats = vector_store.get_stats()
    print(f"\nVector index built successfully!")
    print(f"  - POIs: {stats['pois']}")
    print(f"  - SPUs: {stats['spus']}")


if __name__ == "__main__":
    build_vector_index()
