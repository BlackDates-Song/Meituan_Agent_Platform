"""
FastAPI 后端服务 - 美团外卖推荐与运营分析平台
提供推荐、搜索、运营分析等REST API接口
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sqlite3
from pathlib import Path
import logging
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from backend.api.llm_client import LLMFactory, LLMService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="美团外卖推荐与运营分析平台",
    description="基于大模型的智能推荐和运营分析系统",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据库路径
DB_PATH = project_root / "data" / "db" / "meituan.db"

# 全局LLM服务
llm_service = None

def get_db_connection():
    """获取数据库连接"""
    return sqlite3.connect(DB_PATH)

def init_llm_service():
    """初始化LLM服务"""
    global llm_service
    try:
        client = LLMFactory.create_client('deepseek')
        llm_service = LLMService(client)
        logger.info("LLM服务初始化成功")
    except Exception as e:
        logger.warning(f"LLM服务初始化失败: {e}")
        llm_service = None

# 请求/响应模型
class RecommendRequest(BaseModel):
    user_id: int
    top_k: int = 10
    rec_type: str = "poi"  # poi 或 spu

class SearchRequest(BaseModel):
    query: str
    location: Optional[str] = None
    cuisine: Optional[str] = None
    sort_by: str = "rating"

class OperationAnalysisRequest(BaseModel):
    poi_id: int
    time_range: Optional[str] = "week"

class QuestionRequest(BaseModel):
    poi_id: int
    question: str

# ============= 启动事件 =============
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    logger.info("=" * 60)
    logger.info("美团外卖推荐与运营分析平台启动中...")
    logger.info("=" * 60)
    
    # 检查数据库
    if not DB_PATH.exists():
        logger.error(f"数据库不存在: {DB_PATH}")
        logger.error("请先运行 scripts/load_data.py 加载数据")
    else:
        logger.info(f"✓ 数据库已连接: {DB_PATH}")
    
    # 初始化LLM服务
    init_llm_service()
    
    logger.info("✓ 服务启动完成！")

# ============= 基础接口 =============
@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "美团外卖推荐与运营分析平台 API",
        "version": "1.0.0",
        "endpoints": {
            "推荐": "/api/recommend",
            "搜索": "/api/search",
            "商家详情": "/api/poi/{poi_id}",
            "运营分析": "/api/operation/analysis",
            "智能问答": "/api/qa/answer",
            "统计": "/api/stats"
        }
    }

@app.get("/api/stats")
async def get_statistics():
    """获取平台统计数据"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 获取各项统计
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM pois")
        poi_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM spus")
        spu_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM orders_train")
        order_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "users": user_count,
            "pois": poi_count,
            "spus": spu_count,
            "orders": order_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============= 推荐接口 =============
@app.post("/api/recommend")
async def recommend(request: RecommendRequest):
    """
    推荐接口
    - user_id: 用户ID
    - top_k: 返回推荐数量
    - rec_type: 推荐类型 (poi/spu)
    """
    try:
        conn = get_db_connection()
        
        # 简化版推荐：基于用户历史偏好
        if request.rec_type == "poi":
            # 推荐商家
            query = """
                SELECT p.wm_poi_id, p.wm_poi_name, p.poi_score, 
                       p.delivery_comment_avg_score, p.food_comment_avg_score,
                       COUNT(o.wm_order_id) as popularity
                FROM pois p
                LEFT JOIN orders_train o ON p.wm_poi_id = o.wm_poi_id
                WHERE p.wm_poi_id NOT IN (
                    SELECT wm_poi_id FROM orders_train WHERE user_id = ?
                )
                GROUP BY p.wm_poi_id
                ORDER BY p.poi_score DESC, popularity DESC
                LIMIT ?
            """
        else:
            # 推荐菜品
            query = """
                SELECT s.wm_food_spu_id, s.wm_food_spu_name, s.price,
                       COUNT(os.wm_order_id) as popularity
                FROM spus s
                LEFT JOIN orders_spu_train os ON s.wm_food_spu_id = os.wm_food_spu_id
                GROUP BY s.wm_food_spu_id
                ORDER BY popularity DESC
                LIMIT ?
            """
        
        cursor = conn.cursor()
        if request.rec_type == "poi":
            cursor.execute(query, (request.user_id, request.top_k))
        else:
            cursor.execute(query, (request.top_k,))
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        
        # 如果LLM可用，为每个推荐生成理由
        if llm_service and len(results) > 0:
            try:
                for item in results[:3]:  # 只为前3个生成理由
                    if request.rec_type == "poi":
                        reason = llm_service.generate_recommendation_reason(
                            {"user_id": request.user_id},
                            {"name": item.get("wm_poi_name"), 
                             "score": item.get("poi_score")}
                        )
                    else:
                        reason = f"热门菜品，已有{item.get('popularity', 0)}人购买"
                    item["reason"] = reason
            except Exception as e:
                logger.warning(f"生成推荐理由失败: {e}")
        
        return {
            "user_id": request.user_id,
            "rec_type": request.rec_type,
            "count": len(results),
            "recommendations": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============= 搜索接口 =============
@app.post("/api/search")
async def search_pois(request: SearchRequest):
    """
    智能搜索商家
    支持自然语言查询和结构化搜索
    """
    try:
        conn = get_db_connection()
        
        # 构建SQL查询
        query = """
            SELECT p.wm_poi_id, p.wm_poi_name, p.poi_score,
                   p.delivery_comment_avg_score, p.food_comment_avg_score,
                   p.aor_id, COUNT(o.wm_order_id) as order_count
            FROM pois p
            LEFT JOIN orders_train o ON p.wm_poi_id = o.wm_poi_id
            WHERE 1=1
        """
        params = []
        
        # 如果有位置筛选
        if request.location:
            query += " AND p.aor_id = ?"
            params.append(request.location)
        
        query += " GROUP BY p.wm_poi_id"
        
        # 排序
        if request.sort_by == "rating":
            query += " ORDER BY p.poi_score DESC"
        elif request.sort_by == "popularity":
            query += " ORDER BY order_count DESC"
        
        query += " LIMIT 20"
        
        cursor = conn.cursor()
        cursor.execute(query, params)
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        
        # 如果是自然语言查询，使用LLM解析意图
        intent = None
        if llm_service and request.query:
            try:
                intent_prompt = f"""
解析用户搜索意图，提取关键信息：
查询: {request.query}

请以JSON格式返回：
{{"cuisine": "菜系", "price_range": "价格区间", "rating_min": 最低评分}}
"""
                intent_response = llm_service.client.generate(intent_prompt, max_tokens=100)
                intent = intent_response
            except Exception as e:
                logger.warning(f"意图解析失败: {e}")
        
        return {
            "query": request.query,
            "intent": intent,
            "count": len(results),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============= 商家详情接口 =============
@app.get("/api/poi/{poi_id}")
async def get_poi_detail(poi_id: int):
    """获取商家详细信息"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 商家基本信息
        cursor.execute("""
            SELECT * FROM pois WHERE wm_poi_id = ?
        """, (poi_id,))
        
        poi_data = cursor.fetchone()
        if not poi_data:
            raise HTTPException(status_code=404, detail="商家不存在")
        
        columns = [desc[0] for desc in cursor.description]
        poi_info = dict(zip(columns, poi_data))
        
        # 订单统计
        cursor.execute("""
            SELECT 
                COUNT(*) as total_orders,
                COUNT(DISTINCT user_id) as unique_customers,
                AVG(CASE WHEN ord_period_name = 0 THEN 1 ELSE 0 END) as breakfast_ratio,
                AVG(CASE WHEN ord_period_name = 1 THEN 1 ELSE 0 END) as lunch_ratio,
                AVG(CASE WHEN ord_period_name = 2 THEN 1 ELSE 0 END) as dinner_ratio
            FROM orders_train WHERE wm_poi_id = ?
        """, (poi_id,))
        
        stats = cursor.fetchone()
        stat_columns = [desc[0] for desc in cursor.description]
        poi_stats = dict(zip(stat_columns, stats))
        
        conn.close()
        
        return {
            "info": poi_info,
            "stats": poi_stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============= 运营分析接口 =============
@app.post("/api/operation/analysis")
async def operation_analysis(request: OperationAnalysisRequest):
    """
    商家运营分析
    使用LLM生成运营建议
    """
    try:
        # 获取商家数据
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 商家信息
        cursor.execute("""
            SELECT p.*, 
                   COUNT(o.wm_order_id) as total_orders,
                   COUNT(DISTINCT o.user_id) as unique_customers
            FROM pois p
            LEFT JOIN orders_train o ON p.wm_poi_id = o.wm_poi_id
            WHERE p.wm_poi_id = ?
            GROUP BY p.wm_poi_id
        """, (request.poi_id,))
        
        poi_data = cursor.fetchone()
        if not poi_data:
            raise HTTPException(status_code=404, detail="商家不存在")
        
        columns = [desc[0] for desc in cursor.description]
        merchant_data = dict(zip(columns, poi_data))
        
        conn.close()
        
        # 使用LLM生成运营建议
        advice = None
        if llm_service:
            try:
                advice = llm_service.generate_operation_advice(merchant_data)
            except Exception as e:
                logger.error(f"生成运营建议失败: {e}")
                advice = "LLM服务暂时不可用，无法生成运营建议"
        else:
            advice = "请配置LLM API密钥以启用智能运营建议功能"
        
        return {
            "poi_id": request.poi_id,
            "merchant_data": merchant_data,
            "analysis": advice
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============= 智能问答接口 =============
@app.post("/api/qa/answer")
async def answer_question(request: QuestionRequest):
    """
    基于商家数据的智能问答
    """
    try:
        # 获取商家相关数据作为上下文
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT p.*, COUNT(o.wm_order_id) as total_orders
            FROM pois p
            LEFT JOIN orders_train o ON p.wm_poi_id = o.wm_poi_id
            WHERE p.wm_poi_id = ?
            GROUP BY p.wm_poi_id
        """, (request.poi_id,))
        
        poi_data = cursor.fetchone()
        if not poi_data:
            raise HTTPException(status_code=404, detail="商家不存在")
        
        columns = [desc[0] for desc in cursor.description]
        context_data = dict(zip(columns, poi_data))
        conn.close()
        
        # 构建上下文
        context = f"""
商家信息：
- ID: {context_data['wm_poi_id']}
- 名称: {context_data['wm_poi_name']}
- 综合评分: {context_data['poi_score']}
- 配送评分: {context_data['delivery_comment_avg_score']}
- 菜品评分: {context_data['food_comment_avg_score']}
- 总订单数: {context_data['total_orders']}
"""
        
        # 使用LLM回答问题
        answer = None
        if llm_service:
            try:
                answer = llm_service.answer_question(request.question, context)
            except Exception as e:
                logger.error(f"问答失败: {e}")
                answer = "抱歉，当前无法回答您的问题"
        else:
            answer = "请配置LLM API密钥以启用智能问答功能"
        
        return {
            "poi_id": request.poi_id,
            "question": request.question,
            "answer": answer
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============= 健康检查 =============
@app.get("/health")
async def health_check():
    """健康检查"""
    db_ok = DB_PATH.exists()
    llm_ok = llm_service is not None
    
    return {
        "status": "healthy" if db_ok else "degraded",
        "database": "connected" if db_ok else "disconnected",
        "llm_service": "available" if llm_ok else "unavailable"
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("启动FastAPI服务器...")
    logger.info("访问 http://localhost:8000 查看API")
    logger.info("访问 http://localhost:8000/docs 查看API文档")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
