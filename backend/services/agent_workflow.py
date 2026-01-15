"""
Agent 工作流引擎
支持多Agent协作完成复杂任务
"""

import sqlite3
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.llm_client import LLMFactory


class AgentRole(Enum):
    """Agent角色定义"""
    DATA_ANALYST = "data_analyst"  # 数据分析师
    COMPETITOR_ANALYST = "competitor_analyst"  # 竞品分析师
    OPERATION_ADVISOR = "operation_advisor"  # 运营顾问
    REPORT_GENERATOR = "report_generator"  # 报告生成器


@dataclass
class AgentTask:
    """Agent任务"""
    task_id: str
    role: AgentRole
    description: str
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    status: str = "pending"  # pending, running, completed, failed


class BaseAgent:
    """Agent基类"""
    
    def __init__(self, role: AgentRole, llm_client=None):
        self.role = role
        self.llm_client = llm_client
        if not self.llm_client:
            try:
                self.llm_client = LLMFactory.create_client()
            except:
                pass
    
    def execute(self, task: AgentTask) -> AgentTask:
        """执行任务"""
        raise NotImplementedError


class DataAnalystAgent(BaseAgent):
    """数据分析Agent"""
    
    def __init__(self, db_path: str = "data/db/meituan.db"):
        super().__init__(AgentRole.DATA_ANALYST)
        self.db_path = db_path
    
    def execute(self, task: AgentTask) -> AgentTask:
        """执行数据分析任务"""
        task.status = "running"
        
        poi_id = task.input_data.get("poi_id")
        if not poi_id:
            task.status = "failed"
            task.output_data = {"error": "Missing poi_id"}
            return task
        
        # 获取商家基础数据
        poi_info = self._get_poi_info(poi_id)
        
        # 获取销售数据
        sales_stats = self._get_sales_stats(poi_id)
        
        # 获取热销菜品
        hot_spus = self._get_hot_spus(poi_id)
        
        # 获取用户画像
        user_profile = self._get_user_profile(poi_id)
        
        task.output_data = {
            "poi_info": poi_info,
            "sales_stats": sales_stats,
            "hot_spus": hot_spus,
            "user_profile": user_profile
        }
        task.status = "completed"
        
        return task
    
    def _get_poi_info(self, poi_id: str) -> Dict[str, Any]:
        """获取商家基础信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT wm_poi_id, wm_poi_name, primary_first_tag_name, primary_second_tag_name, 
                   primary_third_tag_name, poi_score, 0, aor_id
            FROM pois
            WHERE wm_poi_id = ?
        """, (poi_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return {}
        
        return {
            "poi_id": str(row[0]),
            "name": row[1] or "",
            "category": f"{row[2] or ''}/{row[3] or ''}/{row[4] or ''}",
            "score": float(row[5]) if row[5] else 0.0,
            "price": float(row[6]) if row[6] else 0.0,
            "district": row[7] or ""
        }
    
    def _get_sales_stats(self, poi_id: str) -> Dict[str, Any]:
        """获取销售统计"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 订单总数
        cursor.execute("""
            SELECT COUNT(*)
            FROM orders_train
            WHERE wm_poi_id = ?
        """, (poi_id,))
        
        row = cursor.fetchone()
        total_orders = row[0] if row[0] else 0
        total_amount = 0.0
        
        # 平均客单价
        avg_order_value = 0
        
        # 独立用户数
        cursor.execute("""
            SELECT COUNT(DISTINCT user_id)
            FROM orders_train
            WHERE wm_poi_id = ?
        """, (poi_id,))
        
        unique_users = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            "total_orders": total_orders,
            "total_amount": round(total_amount, 2),
            "avg_order_value": round(avg_order_value, 2),
            "unique_users": unique_users,
            "avg_orders_per_user": round(total_orders / unique_users, 2) if unique_users > 0 else 0
        }
    
    def _get_hot_spus(self, poi_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """获取热销菜品"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT s.wm_food_spu_id, s.wm_food_spu_name, s.category, COUNT(*) as order_count
            FROM orders_spu_train o
            JOIN spus s ON o.wm_food_spu_id = s.wm_food_spu_id
            JOIN orders_train ot ON o.wm_order_id = ot.wm_order_id
            WHERE ot.wm_poi_id = ?
            GROUP BY s.wm_food_spu_id, s.wm_food_spu_name, s.category
            ORDER BY order_count DESC
            LIMIT ?
        """, (poi_id, top_k))
        
        rows = cursor.fetchall()
        conn.close()
        
        hot_spus = []
        for row in rows:
            hot_spus.append({
                "spu_id": str(row[0]),
                "name": row[1] or "",
                "tag": row[2] or "",
                "order_count": row[3]
            })
        
        return hot_spus
    
    def _get_user_profile(self, poi_id: str) -> Dict[str, Any]:
        """获取用户画像"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取消费用户的消费水平分布
        cursor.execute("""
            SELECT u.avg_pay_amt, COUNT(*) as cnt
            FROM orders_train o
            JOIN users u ON o.user_id = u.user_id
            WHERE o.wm_poi_id = ?
            GROUP BY u.avg_pay_amt
            ORDER BY cnt DESC
        """, (poi_id,))
        
        pay_dist = {}
        for row in cursor.fetchall():
            pay_dist[row[0] or "未知"] = row[1]
        
        # 获取工作日vs周末消费偏好
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN CAST(strftime('%w', datetime(o.order_timestamp, 'unixepoch')) AS INTEGER) IN (0, 6) 
                    THEN '周末' 
                    ELSE '工作日' 
                END as day_type,
                COUNT(*) as cnt
            FROM orders_train o
            WHERE o.wm_poi_id = ?
            GROUP BY day_type
        """, (poi_id,))
        
        day_type_dist = {}
        for row in cursor.fetchall():
            day_type_dist[row[0]] = row[1]
        
        # 获取高峰时段分布
        cursor.execute("""
            SELECT 
                CASE o.ord_period_name
                    WHEN 0 THEN '早餐'
                    WHEN 1 THEN '午餐'
                    WHEN 2 THEN '晚餐'
                    WHEN 3 THEN '夜宵'
                    ELSE '其他'
                END as period,
                COUNT(*) as cnt
            FROM orders_train o
            WHERE o.wm_poi_id = ?
            GROUP BY o.ord_period_name
            ORDER BY cnt DESC
        """, (poi_id,))
        
        period_dist = {}
        for row in cursor.fetchall():
            period_dist[row[0]] = row[1]
        
        conn.close()
        
        return {
            "pay_level_distribution": pay_dist,  # 消费水平分布
            "day_type_distribution": day_type_dist,  # 工作日/周末分布
            "period_distribution": period_dist  # 时段分布
        }


class CompetitorAnalystAgent(BaseAgent):
    """竞品分析Agent"""
    
    def __init__(self, db_path: str = "data/db/meituan.db"):
        super().__init__(AgentRole.COMPETITOR_ANALYST)
        self.db_path = db_path
    
    def execute(self, task: AgentTask) -> AgentTask:
        """执行竞品分析任务"""
        task.status = "running"
        
        poi_id = task.input_data.get("poi_id")
        poi_info = task.input_data.get("poi_info", {})
        
        # 找到同类竞品
        competitors = self._find_competitors(poi_id, poi_info)
        
        # 竞品对比分析
        comparison = self._compare_with_competitors(poi_id, competitors)
        
        task.output_data = {
            "competitors": competitors,
            "comparison": comparison
        }
        task.status = "completed"
        
        return task
    
    def _find_competitors(self, poi_id: str, poi_info: Dict[str, Any], 
                         top_k: int = 5) -> List[Dict[str, Any]]:
        """找到同类竞品"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 同区域、同品类的商家
        category = poi_info.get("category", "").split("/")[0] if poi_info.get("category") else ""
        district = poi_info.get("district", "")
        
        cursor.execute("""
            SELECT wm_poi_id, wm_poi_name, primary_first_tag_name, poi_score, 0 as price
            FROM pois
            WHERE wm_poi_id != ?
              AND COALESCE(aor_id, 0) = ?
              AND primary_first_tag_name = ?
            ORDER BY poi_score DESC
            LIMIT ?
        """, (poi_id, district, category, top_k))
        
        rows = cursor.fetchall()
        conn.close()
        
        competitors = []
        for row in rows:
            competitors.append({
                "poi_id": str(row[0]),
                "name": row[1] or "",
                "category": row[2] or "",
                "score": float(row[3]) if row[3] else 0.0,
                "price": float(row[4]) if row[4] else 0.0
            })
        
        return competitors
    
    def _compare_with_competitors(self, poi_id: str, 
                                  competitors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """与竞品对比"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取当前商家的订单数
        cursor.execute("""
            SELECT COUNT(*) FROM orders_train WHERE wm_poi_id = ?
        """, (poi_id,))
        
        my_orders = cursor.fetchone()[0] or 0
        
        # 获取竞品的订单数
        competitor_orders = []
        for comp in competitors:
            cursor.execute("""
                SELECT COUNT(*) FROM orders_train WHERE wm_poi_id = ?
            """, (comp["poi_id"],))
            
            orders = cursor.fetchone()[0] or 0
            competitor_orders.append({
                "poi_id": comp["poi_id"],
                "name": comp["name"],
                "orders": orders,
                "score": comp["score"],
                "price": comp["price"]
            })
        
        conn.close()
        
        # 计算排名
        all_orders = [my_orders] + [c["orders"] for c in competitor_orders]
        rank = sorted(all_orders, reverse=True).index(my_orders) + 1
        
        return {
            "my_orders": my_orders,
            "rank": rank,
            "total_competitors": len(competitors),
            "competitor_details": competitor_orders
        }


class OperationAdvisorAgent(BaseAgent):
    """运营顾问Agent"""
    
    def __init__(self):
        super().__init__(AgentRole.OPERATION_ADVISOR)
    
    def execute(self, task: AgentTask) -> AgentTask:
        """执行运营建议任务"""
        task.status = "running"
        
        # 综合所有输入数据生成建议
        poi_info = task.input_data.get("poi_info", {})
        sales_stats = task.input_data.get("sales_stats", {})
        hot_spus = task.input_data.get("hot_spus", [])
        comparison = task.input_data.get("comparison", {})
        
        # 基于规则的建议
        suggestions = self._generate_rule_based_suggestions(
            poi_info, sales_stats, hot_spus, comparison
        )
        
        # 基于LLM的深度分析
        if self.llm_client:
            llm_advice = self._generate_llm_advice(
                poi_info, sales_stats, hot_spus, comparison
            )
            suggestions["LLM建议"] = llm_advice
        
        task.output_data = {"suggestions": suggestions}
        task.status = "completed"
        
        return task
    
    def _generate_rule_based_suggestions(self, poi_info, sales_stats, 
                                        hot_spus, comparison) -> Dict[str, Any]:
        """基于规则生成建议"""
        suggestions = {
            "评分优化": [],
            "定价策略": [],
            "产品策略": [],
            "营销策略": []
        }
        
        # 评分优化
        score = poi_info.get("score", 0)
        if score < 4.0:
            suggestions["评分优化"].append(
                "评分低于4.0，建议提升服务质量和菜品品质"
            )
        elif score < 4.5:
            suggestions["评分优化"].append(
                "评分良好，可通过优化就餐体验进一步提升"
            )
        
        # 定价策略
        avg_order = sales_stats.get("avg_order_value", 0)
        if avg_order < 30:
            suggestions["定价策略"].append(
                f"客单价较低（{avg_order:.2f}元），可考虑推出套餐提升客单价"
            )
        
        # 产品策略
        if len(hot_spus) > 0:
            top_dish = hot_spus[0]
            suggestions["产品策略"].append(
                f"主打菜品：{top_dish['name']}，建议作为招牌菜重点推广"
            )
        
        # 竞品对比
        rank = comparison.get("rank", 0)
        if rank > 1:
            suggestions["营销策略"].append(
                f"在同类商家中排名第{rank}，建议加强差异化竞争"
            )
        
        return suggestions
    
    def _generate_llm_advice(self, poi_info, sales_stats, hot_spus, 
                            comparison) -> str:
        """使用LLM生成深度建议"""
        try:
            prompt = f"""作为餐饮运营顾问，请为以下商家提供运营建议：

商家信息：
- 名称：{poi_info.get('name')}
- 类别：{poi_info.get('category')}
- 评分：{poi_info.get('score')}
- 人均：{poi_info.get('price')}元

经营数据：
- 总订单：{sales_stats.get('total_orders')}
- 总营业额：{sales_stats.get('total_amount')}元
- 客单价：{sales_stats.get('avg_order_value')}元
- 独立用户：{sales_stats.get('unique_users')}

热销菜品：
{chr(10).join([f"- {spu['name']} ({spu['order_count']}单)" for spu in hot_spus[:5]])}

竞争情况：
- 同类商家排名：第{comparison.get('rank')}名/{comparison.get('total_competitors')}家

请从以下角度提供具体建议：
1. 如何提升评分和口碑
2. 定价策略优化
3. 菜品结构调整
4. 营销推广方案
5. 差异化竞争策略

请给出切实可行的建议，每条建议控制在50字以内。"""

            response = self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=800
            )
            
            return response
        except Exception as e:
            return f"LLM建议生成失败: {str(e)}"


class ReportGeneratorAgent(BaseAgent):
    """报告生成Agent"""
    
    def __init__(self):
        super().__init__(AgentRole.REPORT_GENERATOR)
    
    def execute(self, task: AgentTask) -> AgentTask:
        """生成运营分析报告"""
        task.status = "running"
        
        report = self._generate_report(task.input_data)
        
        task.output_data = {"report": report}
        task.status = "completed"
        
        return task
    
    def _generate_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成结构化报告"""
        poi_info = data.get("poi_info", {})
        sales_stats = data.get("sales_stats", {})
        hot_spus = data.get("hot_spus", [])
        user_profile = data.get("user_profile", {})
        comparison = data.get("comparison", {})
        suggestions = data.get("suggestions", {})
        
        return {
            "基本信息": {
                "商家名称": poi_info.get("name"),
                "类别": poi_info.get("category"),
                "评分": poi_info.get("score"),
                "人均消费": f"{poi_info.get('price')}元",
                "所在区域": poi_info.get("district")
            },
            "经营数据": {
                "总订单数": sales_stats.get("total_orders"),
                "总营业额": f"{sales_stats.get('total_amount')}元",
                "平均客单价": f"{sales_stats.get('avg_order_value')}元",
                "独立用户数": sales_stats.get("unique_users"),
                "人均订单数": sales_stats.get("avg_orders_per_user")
            },
            "热销菜品TOP10": [
                {
                    "名称": spu.get("name"),
                    "标签": spu.get("tag"),
                    "销量": spu.get("order_count")
                }
                for spu in hot_spus
            ],
            "用户画像": user_profile,
            "竞品分析": {
                "本店订单": comparison.get("my_orders"),
                "市场排名": f"第{comparison.get('rank')}名",
                "竞品数量": comparison.get("total_competitors"),
                "主要竞品": comparison.get("competitor_details", [])[:3]
            },
            "运营建议": suggestions
        }


class AgentWorkflow:
    """Agent工作流编排器"""
    
    def __init__(self, db_path: str = "data/db/meituan.db"):
        self.db_path = db_path
        self.agents = {
            AgentRole.DATA_ANALYST: DataAnalystAgent(db_path),
            AgentRole.COMPETITOR_ANALYST: CompetitorAnalystAgent(db_path),
            AgentRole.OPERATION_ADVISOR: OperationAdvisorAgent(),
            AgentRole.REPORT_GENERATOR: ReportGeneratorAgent()
        }
    
    def run_poi_analysis_workflow(self, poi_id: str) -> Dict[str, Any]:
        """
        运行商家分析工作流
        
        工作流：数据分析 → 竞品分析 → 运营建议 → 报告生成
        """
        # Task 1: 数据分析
        data_task = AgentTask(
            task_id="data_analysis",
            role=AgentRole.DATA_ANALYST,
            description="分析商家数据",
            input_data={"poi_id": poi_id}
        )
        
        data_task = self.agents[AgentRole.DATA_ANALYST].execute(data_task)
        if data_task.status != "completed":
            return {"error": "数据分析失败"}
        
        # Task 2: 竞品分析
        competitor_task = AgentTask(
            task_id="competitor_analysis",
            role=AgentRole.COMPETITOR_ANALYST,
            description="分析竞品情况",
            input_data={
                "poi_id": poi_id,
                "poi_info": data_task.output_data["poi_info"]
            }
        )
        
        competitor_task = self.agents[AgentRole.COMPETITOR_ANALYST].execute(competitor_task)
        
        # Task 3: 运营建议
        advisor_task = AgentTask(
            task_id="operation_advice",
            role=AgentRole.OPERATION_ADVISOR,
            description="生成运营建议",
            input_data={
                "poi_info": data_task.output_data["poi_info"],
                "sales_stats": data_task.output_data["sales_stats"],
                "hot_spus": data_task.output_data["hot_spus"],
                "user_profile": data_task.output_data["user_profile"],
                "comparison": competitor_task.output_data.get("comparison", {})
            }
        )
        
        advisor_task = self.agents[AgentRole.OPERATION_ADVISOR].execute(advisor_task)
        
        # Task 4: 报告生成
        report_task = AgentTask(
            task_id="report_generation",
            role=AgentRole.REPORT_GENERATOR,
            description="生成分析报告",
            input_data={
                **data_task.output_data,
                **competitor_task.output_data,
                **advisor_task.output_data
            }
        )
        
        report_task = self.agents[AgentRole.REPORT_GENERATOR].execute(report_task)
        
        return report_task.output_data


if __name__ == "__main__":
    # 测试工作流
    workflow = AgentWorkflow()
    
    # 使用第一个商家测试
    conn = sqlite3.connect("data/db/meituan.db")
    cursor = conn.cursor()
    cursor.execute("SELECT poi_id FROM pois LIMIT 1")
    poi_id = cursor.fetchone()[0]
    conn.close()
    
    print(f"Running analysis workflow for POI {poi_id}...")
    result = workflow.run_poi_analysis_workflow(str(poi_id))
    
    print("\n分析报告：")
    print(json.dumps(result, ensure_ascii=False, indent=2))
