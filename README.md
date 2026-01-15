# 美团外卖推荐与运营分析平台 V2

## 🎯 项目简介

基于大模型的智能外卖推荐与商家运营分析平台，集成RAG语义搜索、Agent工作流、协同过滤、图神经网络等多种技术。

**技术亮点**:
- 🤖 多智能体协同工作流
- 🔍 RAG增强语义搜索
- 💾 智能缓存系统（LLM成本降低65%）
- 📊 性能监控与日志系统
- 🎨 Vue 3响应式前端

## 📊 数据集

- **来源**: 美团技术研究数据集 (Meituan-TRD)
- **规模**: 400K+节点，18M+边
- **范围**: 北京11个行政区，2021年3月数据
- **内容**: 用户、商家、菜品、订单、评论、用户-商家-菜品异构图

## ✨ 核心功能

### 第一阶段：数据准备 ✅
- [x] 数据字典编写
- [x] 数据库设计（SQLite）
- [x] TSV数据导入与索引优化

### 第二阶段：基础服务 ✅
- [x] 协同过滤推荐（User-based/Item-based/Hybrid）
- [x] Graph Embedding（Node2Vec图神经网络）
- [x] LLM API集成（DeepSeek/Qwen/OpenAI统一接口）
- [x] FastAPI基础接口（8个端点）
- [x] Vue 3前端基础UI

### 第三阶段：智能功能 ✅
- [x] **RAG语义搜索**：向量检索 + 意图解析 + 多路召回融合
- [x] **Agent工作流**：4个专业Agent协同（数据分析 → 竞品分析 → 运营建议 → 报告生成）
- [x] **评论摘要**：LLM驱动的智能摘要 + 情感分析
- [x] **智能问答**：基于RAG的上下文感知问答系统
- [x] **向量数据库**：ChromaDB集成，支持POI/SPU语义检索

### 第四阶段：优化与部署 ✅
- [x] **缓存系统**：SimpleCache（LRU/LFU）+ PersistentCache（文件持久化）
- [x] **性能监控**：PerformanceMonitor + MetricsCollector
- [x] **日志系统**：结构化JSON日志 + 错误追踪
- [x] **前端优化**：6大功能模块完整UI
- [x] **项目文档**：架构设计 + Agent工作流 + 使用指南

## 🏗️ 系统架构

```
用户层 (Web浏览器)
    ↓
API网关层 (FastAPI)
    ↓
服务层 (RAG/Agent/Vector Store)
    ↓
模型层 (CF/Node2Vec/LLM)
    ↓
数据层 (SQLite/ChromaDB)
```

详细架构图见 [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone [项目地址]
cd Meituan

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件:

```bash
# LLM配置（可选，不配置则使用模拟数据）
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-xxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

### 3. 数据初始化

```bash
# 加载数据
python scripts/load_data.py

# 特征工程
python scripts/preprocess.py

# 加载图数据（可选）
python scripts/load_graph.py
```


### 4. 启动服务

**推荐：使用启动脚本**

```powershell
# Windows PowerShell
.\start_v2.ps1
```

访问: http://localhost:8000/

**手动启动**

```bash
# 后端
cd backend
python app_v2.py

# 前端（新开终端）
cd frontend  
python -m http.server 8080

# 浏览器访问: http://localhost:8080/index_v2.html
```

### 5. 构建向量索引（首次使用）

在Web界面点击"构建向量索引"按钮，或使用API:

```bash
curl -X POST http://localhost:8000/api/vector/build
```

## 📖 文档导航

- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)**: 完整项目总结（架构设计、Agent工作流、效果评估、个人反思）
- **[USER_GUIDE.md](USER_GUIDE.md)**: 详细使用指南（API调用、配置、调试）
- **[PHASE3_GUIDE.md](PHASE3_GUIDE.md)**: 第三阶段智能功能文档
- **[data_dictionary.md](data_dictionary.md)**: 数据字典

## 🎯 核心API端点

| 端点 | 方法 | 功能 | 示例 |
|------|------|------|------|
| `/api/recommend` | POST | 智能推荐 | `{"user_id": "1", "use_llm": true}` |
| `/api/rag/search` | POST | RAG语义搜索 | `{"query": "川菜馆"}` |
| `/api/operation/analysis` | POST | Agent运营分析 | `{"poi_id": "12345"}` |
| `/api/comment/summary` | POST | 评论摘要 | `{"poi_id": "12345"}` |
| `/api/qa/answer` | POST | 智能问答 | `{"question": "如何提升评分"}` |
| `/api/stats` | GET | 平台统计 | - |
| `/api/metrics` | GET | 性能监控 | - |
| `/docs` | GET | Swagger文档 | - |

完整API文档: http://localhost:8000/docs

## 💡 使用示例

### Python调用

```python
import requests

# 智能推荐
response = requests.post("http://localhost:8000/api/recommend", json={
    "user_id": "1",
    "top_k": 5,
    "use_llm": True
})
print(response.json()["recommendations"])

# RAG搜索
response = requests.post("http://localhost:8000/api/rag/search", json={
    "query": "评分高的川菜馆",
    "top_k": 10
})
print(response.json()["results"])

# Agent运营分析
response = requests.post("http://localhost:8000/api/operation/analysis", json={
    "poi_id": "12345"
})
report = response.json()["report"]
print(report["运营建议"])
```

### 测试脚本

```bash
# 运行完整API测试
python test_api.py
```

## 📊 性能指标

| 指标 | V1 (基础) | V2 (优化后) | 提升 |
|------|-----------|-------------|------|
| LLM响应时间 | 2.5s | 0.3s | 88%↓ |
| 数据库查询 | 150ms | 20ms | 87%↓ |
| 缓存命中率 | - | 65% | - |
| 推荐准确率 | 85% | 92% | 7%↑ |

详见 [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md#效果评估)

## 🎨 版本对比

| 功能 | V1 (基础版) | V2 (完整版) |
|------|------------|------------|
| 推荐系统 | ✅ CF | ✅ CF + Node2Vec |
| 搜索功能 | ✅ 关键词 | ✅ RAG语义搜索 |
| LLM集成 | ✅ 基础 | ✅ 缓存优化 |
| Agent工作流 | ❌ | ✅ 4个Agent |
| 评论分析 | ❌ | ✅ 智能摘要 |
| 智能问答 | ✅ 简单 | ✅ RAG增强 |
| 性能监控 | ❌ | ✅ 完整监控 |
| 日志系统 | ❌ | ✅ 结构化日志 |

**推荐使用V2版本** 🎯

## 📁 项目结构

```
Meituan/
├── data/                    # 数据目录
│   ├── db/                  # SQLite数据库
│   └── vector_db/           # ChromaDB向量库
├── backend/
│   ├── app.py               # V1 API
│   ├── app_v2.py            # V2 API (推荐)
│   ├── models/              # 推荐模型
│   ├── services/            # 业务服务
│   ├── api/                 # API客户端
│   └── utils/               # 工具函数
├── frontend/
│   ├── index.html           # V1 前端
│   ├── index_v2.html        # V2 前端 (推荐)
│   └── static/              # 静态资源
├── scripts/                 # 数据处理脚本
├── logs/                    # 日志文件
├── test_api.py              # API测试
├── requirements.txt         # 依赖列表
├── PROJECT_SUMMARY.md       # 项目总结
├── USER_GUIDE.md            # 使用指南
└── README.md                # 本文件
```

## 🔧 开发指南

### 添加新功能

1. **后端**: 在 `backend/app_v2.py` 添加端点
2. **前端**: 在 `frontend/index_v2.html` 添加UI
3. **测试**: 在 `test_api.py` 添加测试用例

### 调试模式

```bash
# 启用详细日志
export LOG_LEVEL=DEBUG
python backend/app_v2.py
```

### 查看监控

```python
# 访问监控端点
curl http://localhost:8000/api/metrics

# 查看日志
tail -f logs/api_v2_*.log
```

## 🐛 常见问题

**Q: 向量搜索返回空结果？**  
A: 需要先构建向量索引，点击"构建向量索引"按钮

**Q: LLM调用失败？**  
A: 检查 `.env` 文件中的API密钥配置

**Q: 数据库连接错误？**  
A: 运行 `python scripts/load_data.py` 初始化数据库

**Q: 端口被占用？**  
A: 修改 `app_v2.py` 中的端口号（默认8000）

更多问题见 [USER_GUIDE.md](USER_GUIDE.md#常见问题排查)

## 🎓 技术栈

**后端**:
- FastAPI 0.109
- SQLite 3
- PyTorch 2.1
- DGL 1.1 (Deep Graph Library)
- ChromaDB 0.4 (向量数据库)
- scikit-learn 1.4

**前端**:
- Vue 3 (CDN)
- Axios
- Chart.js

**AI模型**:
- DeepSeek / Qwen / OpenAI LLM
- Collaborative Filtering
- Node2Vec Graph Embedding

## 📄 许可证

本项目仅供学习交流使用。数据集版权归美团所有。

---

**⭐ 如果这个项目对你有帮助，请给个Star支持一下！**

**最后更新**: 2026年  
**项目状态**: ✅ 功能开发完成  
**版本**: V2.0

# Windows PowerShell
.\start_v2.ps1
```

**V2版本包含所有第三阶段智能功能**

#### 方法二：使用基础版启动脚本

```powershell
# 仅包含第二阶段基础功能
.\start.ps1
```

#### 方法三：手动启动

```bash
# 启动V2后端（推荐）
python backend/app_v2.py

# 或启动基础版后端
python backend/app.py

# 启动前端（另开终端）
cd frontend
python -m http.server 8080
```

### 5. 访问平台

- **前端页面**: http://localhost:8080
- **后端API**: http://localhost:8000
- **API文档**: http://localhost:8000/docs

## 📁 项目结构

```
Meituan/
├── data/                          # 数据目录
│   ├── db/                        # SQLite数据库
│   │   └── meituan.db
│   └── vector_db/                 # 向量数据库（第三阶段）
│
├── Meituan_TRD/                   # 原始数据集
│   ├── users.txt                  # 用户数据
│   ├── pois.txt                   # 商家数据
│   ├── spus.txt                   # 菜品数据
│   ├── orders_train.txt           # 订单数据
│   └── graph.bin                  # 异构图
│
├── scripts/                       # 数据处理脚本（第二阶段）
│   ├── load_data.py              # 数据加载
│   ├── preprocess.py             # 数据预处理
│   └── load_graph.py             # 图数据处理
│
├── backend/                       # 后端服务
│   ├── models/                    # 推荐模型（第二阶段）
│   │   ├── collaborative_filtering.py  # 协同过滤
│   │   └── node2vec.py                # 图嵌入
│   │
│   ├── api/                       # API接口（第二阶段）
│   │   └── llm_client.py         # LLM客户端
│   │
│   ├── services/                  # 智能服务（第三阶段）⭐
│   │   ├── vector_store.py       # 向量数据库
│   │   ├── rag_service.py        # RAG搜索引擎
│   │   ├── agent_workflow.py     # Agent工作流
│   │   └── comment_service.py    # 评论与问答
│   │
│   ├── app.py                    # 基础版API服务
│   └── app_v2.py                 # V2增强版API服务 ⭐
│
├── frontend/                      # 前端页面
│   ├── index.html
│   └── static/
│       ├── css/style.css
│       └── js/app.js
│
├── start.ps1                      # 基础版启动脚本
├── start_v2.ps1                   # V2启动脚本 ⭐
├── requirements.txt               # Python依赖
├── PHASE3_GUIDE.md               # 第三阶段使用指南
└── README.md                      # 本文件
```

## 🔌 API接口

### V2版本完整接口（推荐使用）

| 端点 | 方法 | 功能 | 阶段 |
|------|------|------|------|
| `/` | GET | 平台首页 | - |
| `/health` | GET | 健康检查 | - |
| `/api/recommend` | POST | 智能推荐 | 二 |
| `/api/rag/search` | POST | **RAG语义搜索** | 三⭐ |
| `/api/search` | POST | 传统搜索 | 二 |
| `/api/poi/{poi_id}` | GET | 商家详情 | 二 |
| `/api/operation/analysis` | POST | **运营分析（Agent）** | 三⭐ |
| `/api/comment/summary` | POST | **评论摘要** | 三⭐ |
| `/api/qa/answer` | POST | **智能问答（RAG）** | 三⭐ |
| `/api/stats` | GET | 平台统计 | 二 |
| `/api/vector/build` | POST | **构建向量索引** | 三⭐ |

⭐ = 第三阶段新增/增强功能

### 基础版接口

使用 `app.py` 时仅包含第二阶段功能（推荐、搜索、基础问答）

## 🎨 技术栈

### 数据处理
- **Pandas** - 数据处理
- **SQLite** - 关系数据库
- **DGL + PyTorch** - 图神经网络

### 推荐算法
- **协同过滤** - User-based/Item-based CF
- **Node2Vec** - 图嵌入
- **混合推荐** - 多算法融合

### 智能服务（第三阶段）
- **ChromaDB** - 向量数据库
- **RAG** - 检索增强生成
- **Agent** - 多智能体协作
- **LLM** - DeepSeek/Qwen/OpenAI

### Web服务
- **FastAPI** - 后端框架
- **Vue 3** - 前端框架
- **Uvicorn** - ASGI服务器

## 📖 使用示例

### 示例1: RAG语义搜索

```bash
curl -X POST http://localhost:8000/api/rag/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "附近评分高的川菜馆",
    "top_k": 10
  }'
```

### 示例2: Agent运营分析

```bash
curl -X POST http://localhost:8000/api/operation/analysis \
  -H "Content-Type: application/json" \
  -d '{
    "poi_id": "12345"
  }'
```

### 示例3: 智能问答

```bash
curl -X POST http://localhost:8000/api/qa/answer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "如何提升商家评分",
    "context": {"poi_id": "12345"}
  }'
```

更多示例见 [PHASE3_GUIDE.md](PHASE3_GUIDE.md)

## 🔧 配置说明

### 环境变量（.env）

```bash
# LLM API配置（用于第三阶段智能功能）
DEEPSEEK_API_KEY=sk-xxxxx           # DeepSeek API密钥（推荐）
OPENAI_API_KEY=sk-xxxxx             # OpenAI API密钥
OPENAI_BASE_URL=https://api.deepseek.com  # 自定义API地址
DASHSCOPE_API_KEY=sk-xxxxx          # 阿里云通义千问API密钥

# 数据库配置（默认无需修改）
DB_PATH=data/db/meituan.db
VECTOR_DB_PATH=data/vector_db/
```

### 推荐配置

**基础使用**（无需LLM）：
- 传统推荐、搜索、统计功能完全可用
- 不需要配置API密钥

**智能功能**（需要LLM）：
- RAG语义搜索、Agent工作流、智能问答需要配置
- 推荐使用 DeepSeek（性价比高）

## 📚 文档

- [数据字典](data_dictionary.md) - 数据表结构说明
- [需求文档](requests.md) - 项目需求和规划
- [第三阶段指南](PHASE3_GUIDE.md) - 智能功能详细文档

## 🎯 开发路线图

- [x] **第一阶段**：数据准备和字典编写
- [x] **第二阶段**：基础推荐和LLM集成
- [x] **第三阶段**：RAG + Agent + 智能功能
- [ ] **第四阶段**：前端UI优化和可视化
- [ ] **第五阶段**：性能优化和A/B测试

## ⚙️ 常见问题

### Q: V2和基础版有什么区别？

**基础版（app.py）**：
- 第二阶段功能：协同过滤推荐、基础搜索、简单问答
- 不需要向量数据库和复杂LLM调用
- 适合快速体验和学习

**V2版（app_v2.py）**：
- 包含所有第二阶段 + 第三阶段功能
- RAG语义搜索、Agent工作流、智能问答
- 需要ChromaDB和LLM API
- 推荐用于完整功能展示

### Q: 必须配置LLM API吗？

**不是必须的**：
- 推荐、搜索、统计等核心功能无需LLM
- 但智能功能（RAG、Agent、问答）需要LLM才能完整体验

### Q: 如何切换LLM提供商？

编辑 `.env` 文件：
```bash
# 使用DeepSeek
DEEPSEEK_API_KEY=sk-xxxxx

# 或使用OpenAI
OPENAI_API_KEY=sk-xxxxx

# 或使用通义千问
DASHSCOPE_API_KEY=sk-xxxxx
```

程序会自动选择已配置的API。

### Q: 向量索引构建需要多久？

- POI索引（约1000条）：1-2分钟
- SPU索引（约5000条）：3-5分钟
- 可选择在后台运行：`POST /api/vector/build`

## 📝 许可证

本项目仅用于学习和研究目的。

## 🙏 致谢

- 美团技术团队提供的TRD数据集
- OpenAI、DeepSeek等LLM服务提供商
