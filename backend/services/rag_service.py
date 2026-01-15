"""
RAG (Retrieval-Augmented Generation) 服务
支持智能搜索、意图解析、多路召回
"""

import re
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from .vector_store import VectorStore
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.llm_client import LLMFactory


class IntentParser:
    """查询意图解析器"""
    
    # 意图模式匹配
    INTENT_PATTERNS = {
        "nearby": r"(附近|周边|周围|离我近)",
        "high_score": r"(评分高|高分|好评|口碑好)",
        "cheap": r"(便宜|实惠|性价比|平价)",
        "expensive": r"(贵|高端|豪华|精致)",
        "category": r"(川菜|湘菜|粤菜|火锅|烧烤|日料|韩餐|西餐|快餐|小吃)",
        "hot": r"(热门|人气|推荐|网红)",
    }
    
    # 地区映射
    DISTRICTS = [
        "朝阳区", "海淀区", "东城区", "西城区", "丰台区", 
        "石景山区", "通州区", "顺义区", "昌平区", "大兴区", "房山区"
    ]
    
    def __init__(self):
        self.llm_service = None
        try:
            llm_client = LLMFactory.create_client()
            if llm_client:
                from api.llm_client import LLMService
                self.llm_service = LLMService(llm_client)
        except:
            pass
    
    def parse(self, query: str) -> Dict[str, Any]:
        """
        解析用户查询意图
        
        Args:
            query: 用户查询文本
            
        Returns:
            意图解析结果
        """
        intent = {
            "original_query": query,
            "intents": [],
            "filters": {},
            "keywords": []
        }
        
        # 基于规则的意图识别
        for intent_name, pattern in self.INTENT_PATTERNS.items():
            if re.search(pattern, query):
                intent["intents"].append(intent_name)
        
        # 提取地区
        for district in self.DISTRICTS:
            if district in query:
                intent["filters"]["district"] = district
                break
        
        # 提取评分要求
        score_match = re.search(r"(\d+\.?\d*)分以上", query)
        if score_match:
            intent["filters"]["min_score"] = float(score_match.group(1))
        elif "high_score" in intent["intents"]:
            intent["filters"]["min_score"] = 4.0
        
        # 提取价格范围
        price_match = re.search(r"(\d+)元以下", query)
        if price_match:
            intent["filters"]["max_price"] = float(price_match.group(1))
        elif "cheap" in intent["intents"]:
            intent["filters"]["max_price"] = 50.0
        elif "expensive" in intent["intents"]:
            intent["filters"]["min_price"] = 100.0
        
        # 使用LLM增强意图解析
        if self.llm_service:
            try:
                enhanced_intent = self._llm_parse_intent(query)
                if enhanced_intent:
                    intent["llm_analysis"] = enhanced_intent
            except:
                pass
        
        return intent
    
    def _llm_parse_intent(self, query: str) -> Optional[str]:
        """使用LLM解析查询意图"""
        prompt = f"""分析以下用户搜索查询的意图，提取关键信息：

查询：{query}

请提取：
1. 用户想要什么类型的商家或菜品
2. 地理位置偏好
3. 价格偏好
4. 评分要求
5. 其他特殊需求

用简洁的语言回答。"""

        try:
            response = self.llm_service.client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response
        except:
            return None


class RAGSearchService:
    """RAG搜索服务"""
    
    def __init__(self, db_path: str = "data/db/meituan.db"):
        self.db_path = db_path
        self.vector_store = VectorStore()
        self.intent_parser = IntentParser()
    
    def search(self, query: str, top_k: int = 20) -> Dict[str, Any]:
        """
        智能搜索（多路召回 + 融合排序）
        
        Args:
            query: 搜索查询
            top_k: 返回结果数量
            
        Returns:
            搜索结果
        """
        # 1. 意图解析
        intent = self.intent_parser.parse(query)
        
        # 2. 多路召回
        results = {
            "vector_search": [],  # 向量召回
            "keyword_search": [],  # 关键词召回
            "filter_search": []  # 过滤召回
        }
        
        # 2.1 向量语义召回
        if self.vector_store.client:
            vector_pois = self.vector_store.search_pois(query, n_results=top_k)
            results["vector_search"] = vector_pois
        
        # 2.2 关键词召回
        keyword_pois = self._keyword_search(query, intent["filters"], top_k)
        results["keyword_search"] = keyword_pois
        
        # 2.3 基于过滤条件的召回
        if intent["filters"]:
            filter_pois = self._filter_search(intent["filters"], top_k)
            results["filter_search"] = filter_pois
        
        # 3. 结果融合和重排序
        merged_results = self._merge_and_rerank(results, intent, top_k)
        
        return {
            "query": query,
            "intent": intent,
            "results": merged_results,
            "total": len(merged_results)
        }
    
    def _keyword_search(self, query: str, filters: Dict[str, Any], 
                       limit: int = 20) -> List[Dict[str, Any]]:
        """关键词搜索"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 构建SQL查询 - 关联订单和菜品表计算平均价格
        sql = """
            SELECT p.wm_poi_id, p.wm_poi_name, p.primary_first_tag_name,
                   p.primary_second_tag_name, p.primary_third_tag_name,
                   p.poi_score, 
                   COALESCE((
                       SELECT AVG(s.price)
                       FROM orders_train ot
                       JOIN orders_spu_train ost ON ot.wm_order_id = ost.wm_order_id
                       JOIN spus s ON ost.wm_food_spu_id = s.wm_food_spu_id
                       WHERE ot.wm_poi_id = p.wm_poi_id
                       LIMIT 100
                   ), 0) as avg_price,
                   COALESCE(p.aor_id, 0) as district
            FROM pois p
            WHERE 1=1
        """
        params = []
        
        # 关键词匹配
        keywords = query.split()
        if keywords:
            keyword_conditions = " OR ".join([
                "p.wm_poi_name LIKE ? OR p.primary_first_tag_name LIKE ? OR p.primary_second_tag_name LIKE ? OR p.primary_third_tag_name LIKE ?"
                for _ in keywords
            ])
            sql += f" AND ({keyword_conditions})"
            for kw in keywords:
                pattern = f"%{kw}%"
                params.extend([pattern, pattern, pattern, pattern])
        
        # 应用过滤条件
        if "district" in filters and filters["district"]:
            sql += " AND p.aor_id = ?"
            params.append(filters["district"])
        
        if "min_score" in filters:
            sql += " AND p.poi_score >= ?"
            params.append(filters["min_score"])
        
        # 先执行查询，然后在Python中过滤价格
        sql += " ORDER BY p.poi_score DESC LIMIT ?"
        params.append(limit * 3)  # 多取一些，优先筛选有价格的
        
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()
        
        pois_with_price = []
        pois_without_price = []
        
        for row in rows:
            price = float(row[6]) if row[6] else 0.0
            
            # 价格过滤
            if "max_price" in filters and price > filters["max_price"] and price > 0:
                continue
            if "min_price" in filters and price < filters["min_price"] and price > 0:
                continue
            
            poi_data = {
                "poi_id": str(row[0]),
                "name": row[1] or "",
                "category": f"{row[2] or ''}/{row[3] or ''}/{row[4] or ''}",
                "score": float(row[5]) if row[5] else 0.0,
                "price": price,
                "district": str(row[7]) if row[7] else "未知",
                "source": "keyword"
            }
            
            # 有价格的优先
            if price > 0:
                pois_with_price.append(poi_data)
            else:
                pois_without_price.append(poi_data)
        
        # 合并结果：有价格的在前
        pois = pois_with_price + pois_without_price
        return pois[:limit]
    
    def _filter_search(self, filters: Dict[str, Any], 
                      limit: int = 20) -> List[Dict[str, Any]]:
        """基于过滤条件的搜索"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 关联订单和菜品表计算平均价格
        sql = """
            SELECT p.wm_poi_id, p.wm_poi_name, p.primary_first_tag_name,
                   p.primary_second_tag_name, p.primary_third_tag_name,
                   p.poi_score,
                   COALESCE((
                       SELECT AVG(s.price)
                       FROM orders_train ot
                       JOIN orders_spu_train ost ON ot.wm_order_id = ost.wm_order_id
                       JOIN spus s ON ost.wm_food_spu_id = s.wm_food_spu_id
                       WHERE ot.wm_poi_id = p.wm_poi_id
                       LIMIT 100
                   ), 0) as avg_price,
                   COALESCE(p.aor_id, 0) as district
            FROM pois p
            WHERE 1=1
        """
        params = []
        
        if "district" in filters and filters["district"]:
            sql += " AND p.aor_id = ?"
            params.append(filters["district"])
        
        if "min_score" in filters:
            sql += " AND p.poi_score >= ?"
            params.append(filters["min_score"])
        
        sql += " ORDER BY p.poi_score DESC LIMIT ?"
        params.append(limit * 3)
        
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()
        
        pois_with_price = []
        pois_without_price = []
        
        for row in rows:
            price = float(row[6]) if row[6] else 0.0
            
            # 价格过滤
            if "max_price" in filters and price > filters["max_price"] and price > 0:
                continue
            if "min_price" in filters and price < filters["min_price"] and price > 0:
                continue
            
            poi_data = {
                "poi_id": str(row[0]),
                "name": row[1] or "",
                "category": f"{row[2] or ''}/{row[3] or ''}/{row[4] or ''}",
                "score": float(row[5]) if row[5] else 0.0,
                "price": price,
                "district": str(row[7]) if row[7] else "未知",
                "source": "filter"
            }
            
            # 有价格的优先
            if price > 0:
                pois_with_price.append(poi_data)
            else:
                pois_without_price.append(poi_data)
        
        # 合并：有价格优先，按价格和评分排序
        if "max_price" in filters or "min_price" in filters:
            pois_with_price = sorted(pois_with_price, key=lambda x: (x['price'], -x['score']))
        
        pois = pois_with_price + pois_without_price
        return pois[:limit]
    
    def _merge_and_rerank(self, results: Dict[str, List], intent: Dict[str, Any],
                         top_k: int) -> List[Dict[str, Any]]:
        """融合多路召回结果并重排序"""
        # 使用字典去重（以poi_id为key）
        poi_map = {}
        
        # 权重配置
        weights = {
            "vector_search": 0.4,
            "keyword_search": 0.4,
            "filter_search": 0.2
        }
        
        # 合并结果
        for source, pois in results.items():
            for i, poi in enumerate(pois):
                poi_id = poi["poi_id"]
                
                # 计算得分（位置越靠前得分越高）
                position_score = (len(pois) - i) / len(pois) if pois else 0
                
                if poi_id not in poi_map:
                    poi_map[poi_id] = poi.copy()
                    poi_map[poi_id]["rank_score"] = 0
                    poi_map[poi_id]["sources"] = []
                
                # 累加加权得分
                poi_map[poi_id]["rank_score"] += weights.get(source, 0.3) * position_score
                poi_map[poi_id]["sources"].append(source)
        
        # 按综合得分排序
        merged = list(poi_map.values())
        merged.sort(key=lambda x: (
            x["rank_score"],  # 综合得分
            x.get("score", 0),  # 商家评分
            -x.get("price", 999)  # 价格（越低越好）
        ), reverse=True)
        
        return merged[:top_k]


if __name__ == "__main__":
    # 测试RAG搜索
    rag = RAGSearchService()
    
    test_queries = [
        "附近评分高的川菜馆",
        "朝阳区100元以下的日料",
        "推荐一些实惠的火锅店"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"查询: {query}")
        print('='*60)
        
        result = rag.search(query, top_k=5)
        
        print(f"\n意图解析: {result['intent']}")
        print(f"\n找到 {result['total']} 个结果:\n")
        
        for i, poi in enumerate(result['results'], 1):
            print(f"{i}. {poi['name']}")
            print(f"   类别: {poi['category']}")
            print(f"   评分: {poi['score']} | 价格: {poi['price']}元 | 区域: {poi['district']}")
            print(f"   来源: {', '.join(poi['sources'])} | 得分: {poi['rank_score']:.3f}")
            print()
