"""
评论摘要与问答服务
基于RAG的评论分析和上下文感知问答
"""

import sqlite3
from typing import List, Dict, Any, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.llm_client import LLMFactory, LLMService
from services.vector_store import VectorStore


class CommentService:
    """评论服务"""
    
    def __init__(self, db_path: str = "data/db/meituan.db"):
        self.db_path = db_path
        self.vector_store = VectorStore()
        
        # 初始化LLM服务
        self.llm_service = None
        try:
            llm_client = LLMFactory.create_client()
            if llm_client:
                self.llm_service = LLMService(llm_client)
        except:
            pass
    
    def get_poi_summary(self, poi_id: str) -> Dict[str, Any]:
        """
        获取商家评论摘要（模拟数据）
        
        Args:
            poi_id: 商家ID
            
        Returns:
            评论摘要信息
        """
        # 获取商家信息
        poi_info = self._get_poi_info(poi_id)
        if not poi_info:
            return {"error": "商家不存在"}
        
        # 获取订单统计作为评论参考
        order_stats = self._get_order_stats(poi_id)
        
        # 生成模拟评论摘要
        summary = self._generate_mock_summary(poi_info, order_stats)
        
        # 如果有LLM，生成增强摘要
        if self.llm_service:
            enhanced_summary = self._generate_llm_summary(poi_info, order_stats, summary)
            summary["llm_summary"] = enhanced_summary
        
        return summary
    
    def answer_question(self, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        基于上下文的问答
        
        Args:
            question: 用户问题
            context: 上下文信息（商家ID、用户ID等）
            
        Returns:
            答案
        """
        if not self.llm_service:
            return "问答服务暂不可用，请配置LLM API密钥"
        
        # 检索相关信息
        relevant_info = self._retrieve_relevant_info(question, context)
        
        # 构建增强的提示词
        prompt = self._build_qa_prompt(question, relevant_info)
        
        # 调用LLM生成答案
        try:
            answer = self.llm_service.client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            return answer
        except Exception as e:
            return f"问答失败: {str(e)}"
    
    def _get_poi_info(self, poi_id: str) -> Optional[Dict[str, Any]]:
        """获取商家信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT wm_poi_id, wm_poi_name, primary_first_tag_name, primary_second_tag_name, 
                   poi_score, aor_id
            FROM pois
            WHERE wm_poi_id = ?
        """, (poi_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return {
            "poi_id": str(row[0]),
            "name": row[1] or "",
            "category": f"{row[2] or ''}/{row[3] or ''}",
            "score": float(row[4]) if row[4] else 0.0,
            "district": str(row[5]) if row[5] is not None else ""
        }
    
    def _get_order_stats(self, poi_id: str) -> Dict[str, Any]:
        """获取订单统计"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取订单数和用户数
        cursor.execute("""
            SELECT COUNT(DISTINCT wm_order_id) as order_count,
                   COUNT(DISTINCT user_id) as user_count
            FROM orders_train
            WHERE wm_poi_id = ?
        """, (poi_id,))
        
        row = cursor.fetchone()
        
        total_orders = row[0] if row and row[0] else 0
        unique_users = row[1] if row and row[1] else 0
        
        # 计算平均客单价（基于订单价格区间估算）
        cursor.execute("""
            SELECT order_price_interval, COUNT(*) as count
            FROM orders_train
            WHERE wm_poi_id = ?
            GROUP BY order_price_interval
        """, (poi_id,))
        
        price_intervals = cursor.fetchall()
        conn.close()
        
        # 估算平均客单价
        avg_price = self._estimate_avg_price(price_intervals, total_orders)
        
        return {
            "total_orders": total_orders,
            "unique_users": unique_users,
            "avg_order_value": avg_price
        }
    
    def _estimate_avg_price(self, price_intervals, total_orders):
        """根据价格区间估算平均价格"""
        if total_orders == 0:
            return 0.0
        
        # 价格区间映射：<29, [29,36), [36,49), [49,65), >=65
        interval_map = {
            "<29": 24.5,  # 取20-29的中点
            "[29,36)": 32.5,
            "[36,49)": 42.5,
            "[49,65)": 57.0,
            ">=65": 75.0  # 估算值
        }
        
        total_value = 0.0
        for interval, count in price_intervals:
            price = interval_map.get(interval, 50.0)  # 默认50元
            total_value += price * count
        
        return round(total_value / total_orders, 1) if total_orders > 0 else 0.0
    
    def _generate_mock_summary(self, poi_info: Dict[str, Any], 
                              order_stats: Dict[str, Any]) -> Dict[str, Any]:
        """生成模拟评论摘要"""
        score = poi_info.get("score", 0)
        total_orders = order_stats.get("total_orders", 0)
        
        # 基于评分生成正面/负面评价
        positive_aspects = []
        negative_aspects = []
        
        if score >= 4.5:
            positive_aspects = ["菜品美味", "服务热情", "环境舒适", "性价比高"]
            negative_aspects = ["偶尔需要等位"]
        elif score >= 4.0:
            positive_aspects = ["口味不错", "分量足", "位置方便"]
            negative_aspects = ["高峰期服务有点慢", "环境一般"]
        elif score >= 3.5:
            positive_aspects = ["价格实惠", "口味还行"]
            negative_aspects = ["服务态度需改进", "环境较差", "菜品质量不稳定"]
        else:
            positive_aspects = ["价格便宜"]
            negative_aspects = ["口味一般", "服务差", "环境差", "不推荐"]
        
        # 生成情感分布
        positive_ratio = min(score / 5.0, 1.0)
        
        return {
            "poi_id": poi_info["poi_id"],
            "poi_name": poi_info["name"],
            "total_reviews": total_orders,  # 用订单数模拟评论数
            "average_score": score,
            "sentiment": {
                "positive": round(positive_ratio * 100, 1),
                "neutral": round((1 - positive_ratio) * 50, 1),
                "negative": round((1 - positive_ratio) * 50, 1)
            },
            "positive_aspects": positive_aspects,
            "negative_aspects": negative_aspects,
            "keywords": ["美味", "服务", "环境", "性价比"]
        }
    
    def _generate_llm_summary(self, poi_info: Dict[str, Any], 
                             order_stats: Dict[str, Any],
                             mock_summary: Dict[str, Any]) -> str:
        """使用LLM生成评论摘要"""
        try:
            prompt = f"""基于以下商家信息和数据，生成一段评论摘要（150字以内）：

商家：{poi_info['name']}
类别：{poi_info['category']}
评分：{poi_info['score']}/5.0
订单数：{order_stats['total_orders']}

用户反馈：
优点：{', '.join(mock_summary['positive_aspects'])}
缺点：{', '.join(mock_summary['negative_aspects'])}

请用客观、简洁的语言总结该商家的整体表现和用户评价。"""

            summary = self.llm_service.client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300
            )
            
            return summary
        except Exception as e:
            return f"摘要生成失败: {str(e)}"
    
    def _retrieve_relevant_info(self, question: str, 
                               context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """检索相关信息"""
        relevant_info = {
            "question": question,
            "context": context or {}
        }
        
        # 如果问题涉及商家，使用向量搜索找相关商家
        if any(kw in question for kw in ["商家", "餐厅", "店", "推荐"]):
            if self.vector_store.client:
                pois = self.vector_store.search_pois(question, n_results=3)
                relevant_info["related_pois"] = pois
        
        # 如果问题涉及菜品
        if any(kw in question for kw in ["菜", "菜品", "食物", "吃"]):
            if self.vector_store.client:
                spus = self.vector_store.search_spus(question, n_results=5)
                relevant_info["related_spus"] = spus
        
        # 如果有特定商家上下文
        if context and "poi_id" in context:
            poi_info = self._get_poi_info(context["poi_id"])
            if poi_info:
                relevant_info["poi_info"] = poi_info
                order_stats = self._get_order_stats(context["poi_id"])
                relevant_info["order_stats"] = order_stats
        
        return relevant_info
    
    def _build_qa_prompt(self, question: str, 
                        relevant_info: Dict[str, Any]) -> str:
        """构建问答提示词"""
        context_parts = [f"用户问题：{question}\n"]
        
        # 添加相关商家信息
        if "related_pois" in relevant_info and relevant_info["related_pois"]:
            context_parts.append("\n相关商家：")
            for poi in relevant_info["related_pois"][:3]:
                context_parts.append(
                    f"- {poi['name']} | {poi['category']} | "
                    f"评分{poi['score']} | {poi['district']}"
                )
        
        # 添加相关菜品信息
        if "related_spus" in relevant_info and relevant_info["related_spus"]:
            context_parts.append("\n相关菜品：")
            for spu in relevant_info["related_spus"][:5]:
                context_parts.append(f"- {spu['name']} ({spu['category']})")
        
        # 添加特定商家信息
        if "poi_info" in relevant_info:
            poi = relevant_info["poi_info"]
            stats = relevant_info.get("order_stats", {})
            context_parts.append(f"\n商家详情：")
            context_parts.append(f"- 名称：{poi['name']}")
            context_parts.append(f"- 类别：{poi['category']}")
            context_parts.append(f"- 评分：{poi['score']}")
            context_parts.append(f"- 人均消费：约{stats.get('avg_order_value', 0)}元")
            context_parts.append(f"- 订单数：{stats.get('total_orders', 0)}")
        
        context_parts.append("\n请基于以上信息，简洁专业地回答用户问题（200字以内）。")
        
        return "\n".join(context_parts)


class KnowledgeQA:
    """知识问答系统"""
    
    def __init__(self):
        self.comment_service = CommentService()
        self.common_questions = {
            "如何提升商家评分": "提升商家评分的关键因素：1) 保证菜品质量和口味稳定性 2) 提升服务质量和响应速度 3) 改善就餐环境 4) 控制价格合理性 5) 积极处理用户反馈",
            "什么是客单价": "客单价是指每位顾客平均消费金额，计算公式为：总营业额÷订单数。提升客单价的方法包括推出套餐、关联销售、会员优惠等。",
            "如何分析竞品": "竞品分析关注：1) 同区域同品类商家 2) 对比评分、价格、订单量 3) 分析差异化优势 4) 学习优秀经验 5) 找到自身定位",
        }
    
    def answer(self, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        回答用户问题
        
        Args:
            question: 问题
            context: 上下文
            
        Returns:
            答案
        """
        # 先检查常见问题
        for q, a in self.common_questions.items():
            if q in question:
                return a
        
        # 使用RAG问答
        return self.comment_service.answer_question(question, context)


if __name__ == "__main__":
    # 测试评论服务
    service = CommentService()
    
    # 测试商家摘要
    conn = sqlite3.connect("data/db/meituan.db")
    cursor = conn.cursor()
    cursor.execute("SELECT wm_poi_id FROM pois LIMIT 1")
    poi_id = cursor.fetchone()[0]
    conn.close()
    
    print("="*60)
    print("商家评论摘要测试")
    print("="*60)
    
    summary = service.get_poi_summary(str(poi_id))
    print(f"\n商家：{summary.get('poi_name')}")
    print(f"评分：{summary.get('average_score')}")
    print(f"评论数：{summary.get('total_reviews')}")
    print(f"\n优点：{', '.join(summary.get('positive_aspects', []))}")
    print(f"缺点：{', '.join(summary.get('negative_aspects', []))}")
    
    if "llm_summary" in summary:
        print(f"\nLLM摘要：\n{summary['llm_summary']}")
    
    # 测试问答
    print("\n" + "="*60)
    print("智能问答测试")
    print("="*60)
    
    qa = KnowledgeQA()
    
    test_questions = [
        "推荐一些评分高的川菜馆",
        "如何提升商家评分",
        f"这个商家{poi_id}怎么样？"
    ]
    
    for q in test_questions:
        print(f"\n问：{q}")
        context = {"poi_id": str(poi_id)} if "商家" in q and poi_id else None
        answer = qa.answer(q, context)
        print(f"答：{answer}")
