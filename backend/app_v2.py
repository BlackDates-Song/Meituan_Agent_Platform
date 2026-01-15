"""
美团外卖推荐平台 V2 - 集成第三阶段智能功能
包含RAG搜索、Agent工作流、评论摘要与问答
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sqlite3
from pathlib import Path
import sys
import os

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from api.llm_client import LLMFactory, LLMService
from services.vector_store import VectorStore
from services.rag_service import RAGSearchService
from services.agent_workflow import AgentWorkflow
from services.comment_service import CommentService, KnowledgeQA
from utils.logging_config import setup_logger, RequestLogger, metrics_collector, track_metric

# 配置日志
logger = setup_logger("api_v2")

# 创建FastAPI应用
app = FastAPI(
    title="美团外卖推荐与运营分析平台 V2",
    description="集成RAG、Agent工作流的智能推荐和运营分析系统",
    version="2.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    return await RequestLogger(logger)(request, call_next)

# 数据库路径
DB_PATH = "data/db/meituan.db"

# 全局服务
llm_service = None
vector_store = None
rag_service = None
agent_workflow = None
comment_service = None
qa_system = None


# ==================== 请求/响应模型 ====================

class RecommendRequest(BaseModel):
    user_id: Optional[str] = None
    spu_id: Optional[str] = None
    top_k: int = 10
    use_llm: bool = False

class SearchRequest(BaseModel):
    query: str
    district: Optional[str] = None
    category: Optional[str] = None
    min_score: Optional[float] = None
    max_price: Optional[float] = None
    top_k: int = 20

class OperationAnalysisRequest(BaseModel):
    poi_id: str

class QARequest(BaseModel):
    question: str
    context: Optional[Dict[str, Any]] = None

class RAGSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 20

class CommentSummaryRequest(BaseModel):
    poi_id: str


# ==================== 启动事件 ====================

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    global llm_service, vector_store, rag_service, agent_workflow, comment_service, qa_system
    
    logger.info("="*60)
    logger.info("美团外卖推荐与运营分析平台 V2 启动中...")
    logger.info("="*60)
    
    # 检查数据库
    if not Path(DB_PATH).exists():
        logger.error(f"数据库不存在: {DB_PATH}")
        logger.error("请先运行: python scripts/load_data.py")
    else:
        logger.info(f"✓ 数据库已连接: {DB_PATH}")
    
    # 初始化LLM服务
    try:
        llm_client = LLMFactory.create_client()
        if llm_client:
            llm_service = LLMService(llm_client)
            logger.info("✓ LLM服务已初始化")
    except Exception as e:
        logger.warning(f"⚠ LLM服务初始化失败: {e}")
    
    # 初始化向量存储
    try:
        vector_store = VectorStore()
        logger.info("✓ 向量存储已初始化")
    except Exception as e:
        logger.warning(f"⚠ 向量存储初始化失败: {e}")
    
    # 初始化RAG服务
    try:
        rag_service = RAGSearchService(DB_PATH)
        logger.info("✓ RAG搜索服务已初始化")
    except Exception as e:
        logger.warning(f"⚠ RAG搜索服务初始化失败: {e}")
    
    # 初始化Agent工作流
    try:
        agent_workflow = AgentWorkflow(DB_PATH)
        logger.info("✓ Agent工作流已初始化")
    except Exception as e:
        logger.warning(f"⚠ Agent工作流初始化失败: {e}")
    
    # 初始化评论和问答服务
    try:
        comment_service = CommentService(DB_PATH)
        qa_system = KnowledgeQA()
        logger.info("✓ 评论与问答服务已初始化")
    except Exception as e:
        logger.warning(f"⚠ 评论与问答服务初始化失败: {e}")
    
    logger.info("="*60)
    logger.info("✓ 平台启动完成！")
    logger.info("  - API文档: http://localhost:8000/docs")
    logger.info("  - 前端页面: http://localhost:8080")
    logger.info("="*60)


# ==================== 基础接口 ====================

@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "美团外卖推荐与运营分析平台",
        "version": "2.0.0",
        "features": [
            "智能推荐（协同过滤 + Node2Vec）",
            "RAG语义搜索（向量检索 + 意图解析）",
            "Agent工作流（运营分析 + 竞品对比）",
            "评论摘要与智能问答",
            "LLM增强服务"
        ],
        "endpoints": {
            "推荐": "POST /api/recommend",
            "RAG搜索": "POST /api/rag/search",
            "传统搜索": "POST /api/search",
            "商家详情": "GET /api/poi/{poi_id}",
            "运营分析": "POST /api/operation/analysis",
            "评论摘要": "POST /api/comment/summary",
            "智能问答": "POST /api/qa/answer",
            "平台统计": "GET /api/stats",
            "构建向量索引": "POST /api/vector/build"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    db_ok = Path(DB_PATH).exists()
    
    return {
        "status": "healthy" if db_ok else "degraded",
        "database": "connected" if db_ok else "disconnected",
        "llm_service": "available" if llm_service else "unavailable",
        "vector_store": "available" if vector_store and vector_store.client else "unavailable",
        "rag_service": "available" if rag_service else "unavailable",
        "agent_workflow": "available" if agent_workflow else "unavailable"
    }


# ==================== 推荐接口 ====================

@app.post("/api/recommend")
@track_metric("api.recommend")
async def recommend(request: RecommendRequest):
    """智能推荐接口"""
    logger.info(f"推荐请求: user_id={request.user_id}, use_llm={request.use_llm}")
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        recommendations = []
        
        if request.user_id:
            # 基于用户的推荐
            cursor.execute("""
                SELECT p.wm_poi_id, p.wm_poi_name, p.primary_first_tag_name, p.poi_score, 0, p.aor_id
                FROM pois p
                WHERE p.wm_poi_id NOT IN (
                    SELECT DISTINCT wm_poi_id FROM orders_train WHERE user_id = ?
                )
                ORDER BY p.poi_score DESC
                LIMIT ?
            """, (request.user_id, request.top_k))
            
            rows = cursor.fetchall()
            for row in rows:
                recommendations.append({
                    "poi_id": str(row[0]),
                    "name": row[1] or "",
                    "category": row[2] or "",
                    "score": float(row[3]) if row[3] else 0.0,
                    "price": float(row[4]) if row[4] else 0.0,
                    "district": row[5] or ""
                })
        
        elif request.spu_id:
            # 基于菜品的商家推荐
            cursor.execute("""
                SELECT DISTINCT p.wm_poi_id, p.wm_poi_name, p.primary_first_tag_name, p.poi_score, 0, p.aor_id
                FROM pois p
                JOIN orders_train o ON p.wm_poi_id = o.wm_poi_id
                WHERE o.wm_order_id IN (
                    SELECT DISTINCT wm_order_id FROM orders_spu_train 
                    WHERE wm_food_spu_id IN (
                        SELECT wm_food_spu_id FROM spus WHERE category = (
                            SELECT category FROM spus WHERE wm_food_spu_id = ?
                        )
                    )
                )
                ORDER BY p.poi_score DESC
                LIMIT ?
            """, (request.spu_id, request.top_k))
            
            rows = cursor.fetchall()
            for row in rows:
                recommendations.append({
                    "poi_id": str(row[0]),
                    "name": row[1] or "",
                    "category": row[2] or "",
                    "score": float(row[3]) if row[3] else 0.0,
                    "price": float(row[4]) if row[4] else 0.0,
                    "district": row[5] or ""
                })
        
        conn.close()
        
        # 使用LLM生成推荐理由
        if request.use_llm and llm_service and recommendations:
            try:
                for rec in recommendations[:3]:
                    reason = llm_service.generate_recommendation_reason(
                        {"user_id": request.user_id or "unknown"},
                        rec
                    )
                    rec["llm_reason"] = reason
            except Exception as e:
                logger.warning(f"LLM推荐理由生成失败: {e}")
        
        return {
            "recommendations": recommendations,
            "count": len(recommendations)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== RAG搜索接口 ====================

@app.post("/api/rag/search")
async def rag_search(request: RAGSearchRequest):
    """RAG智能搜索（向量检索 + 意图解析 + 多路召回）"""
    try:
        if not rag_service:
            raise HTTPException(status_code=503, detail="RAG搜索服务未初始化")
        
        result = rag_service.search(request.query, request.top_k)
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 传统搜索接口 ====================

@app.post("/api/search")
async def search_pois(request: SearchRequest):
    """传统搜索（基于SQL筛选）"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        sql = """
            SELECT wm_poi_id, wm_poi_name, primary_first_tag_name, primary_second_tag_name, 
                   poi_score, 0, aor_id
            FROM pois
            WHERE 1=1
        """
        params = []
        
        if request.query:
            sql += " AND (wm_poi_name LIKE ? OR primary_first_tag_name LIKE ? OR primary_second_tag_name LIKE ?)"
            pattern = f"%{request.query}%"
            params.extend([pattern, pattern, pattern])
        
        if request.district:
            sql += " AND aor_id = ?"
            params.append(request.district)
        
        if request.category:
            sql += " AND (poi_cate1 = ? OR poi_cate2 = ?)"
            params.extend([request.category, request.category])
        
        if request.min_score:
            sql += " AND score >= ?"
            params.append(request.min_score)
        
        if request.max_price:
            sql += " AND price <= ?"
            params.append(request.max_price)
        
        sql += " ORDER BY score DESC LIMIT ?"
        params.append(request.top_k)
        
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            results.append({
                "poi_id": str(row[0]),
                "name": row[1] or "",
                "category": f"{row[2] or ''}/{row[3] or ''}",
                "score": float(row[4]) if row[4] else 0.0,
                "price": float(row[5]) if row[5] else 0.0,
                "district": row[6] or ""
            })
        
        return {
            "query": request.query,
            "results": results,
            "count": len(results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 商家详情接口 ====================

@app.get("/api/poi/{poi_id}")
async def get_poi_detail(poi_id: str):
    """获取商家详细信息"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 商家基本信息
        cursor.execute("""
            SELECT wm_poi_id, wm_poi_name, primary_first_tag_name, primary_second_tag_name, 
                   primary_third_tag_name, poi_score, 0, aor_id
            FROM pois
            WHERE wm_poi_id = ?
        """, (poi_id,))
        
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="商家不存在")
        
        poi_info = {
            "poi_id": str(row[0]),
            "name": row[1] or "",
            "category": f"{row[2] or ''}/{row[3] or ''}/{row[4] or ''}",
            "score": float(row[5]) if row[5] else 0.0,
            "price": float(row[6]) if row[6] else 0.0,
            "district": row[7] or ""
        }
        
        # 订单统计
        cursor.execute("""
            SELECT COUNT(*), COUNT(DISTINCT user_id)
            FROM orders_train
            WHERE wm_poi_id = ?
        """, (poi_id,))
        
        stats_row = cursor.fetchone()
        poi_stats = {
            "total_orders": stats_row[0] or 0,
            "total_amount": 0.0,
            "unique_customers": stats_row[1] or 0
        }
        
        conn.close()
        
        return {
            "info": poi_info,
            "stats": poi_stats
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 运营分析接口（Agent工作流）====================

@app.post("/api/operation/analysis")
async def operation_analysis(request: OperationAnalysisRequest):
    """商家运营分析（使用Agent工作流）"""
    try:
        if not agent_workflow:
            raise HTTPException(status_code=503, detail="Agent工作流服务未初始化")
        
        result = agent_workflow.run_poi_analysis_workflow(request.poi_id)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return {
            "poi_id": request.poi_id,
            "report": result.get("report", {}),
            "status": "completed"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 评论摘要接口 ====================

@app.post("/api/comment/summary")
async def get_comment_summary(request: CommentSummaryRequest):
    """获取商家评论摘要"""
    try:
        if not comment_service:
            raise HTTPException(status_code=503, detail="评论服务未初始化")
        
        summary = comment_service.get_poi_summary(request.poi_id)
        
        if "error" in summary:
            raise HTTPException(status_code=404, detail=summary["error"])
        
        return summary
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 智能问答接口 ====================

@app.post("/api/qa/answer")
async def answer_question(request: QARequest):
    """智能问答（基于RAG）"""
    try:
        if not qa_system:
            raise HTTPException(status_code=503, detail="问答服务未初始化")
        
        answer = qa_system.answer(request.question, request.context)
        
        return {
            "question": request.question,
            "answer": answer,
            "context": request.context
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 平台统计接口 ====================

@app.get("/api/stats")
async def get_stats():
    """获取平台统计数据"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM pois")
        total_pois = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM spus")
        total_spus = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM orders_train")
        total_orders = cursor.fetchone()[0]
        
        conn.close()
        
        # 向量库统计
        vector_stats = {"pois": 0, "spus": 0, "comments": 0}
        if vector_store:
            try:
                vector_stats = vector_store.get_stats()
            except:
                pass
        
        return {
            "total_users": total_users,
            "total_pois": total_pois,
            "total_spus": total_spus,
            "total_orders": total_orders,
            "vector_indexed": vector_stats
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 向量索引构建接口 ====================

@app.post("/api/vector/build")
async def build_vector_index():
    """构建向量索引"""
    try:
        if not vector_store:
            raise HTTPException(status_code=503, detail="向量存储未初始化")
        
        # 索引商家和菜品
        vector_store.index_pois(DB_PATH)
        vector_store.index_spus(DB_PATH)
        
        stats = vector_store.get_stats()
        
        return {
            "status": "success",
            "message": "向量索引构建完成",
            "stats": stats
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 监控接口 ====================

@app.get("/api/metrics")
async def get_metrics():
    """获取性能指标"""
    from utils.cache import perf_monitor, llm_cache
    
    return {
        "performance": perf_monitor.get_stats(),
        "cache": {
            "llm_cache": llm_cache.get_stats() if llm_cache else {}
        },
        "api_metrics": metrics_collector.export_all()
    }


# ==================== 主程序 ====================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("="*60)
    logger.info("启动美团外卖推荐与运营分析平台 V2...")
    logger.info("="*60)
    logger.info("  API地址: http://localhost:8000")
    logger.info("  API文档: http://localhost:8000/docs")
    logger.info("  前端页面: http://localhost:8080")
    logger.info("="*60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
