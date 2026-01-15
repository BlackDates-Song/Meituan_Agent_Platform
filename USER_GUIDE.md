# 美团外卖推荐平台 - 完整使用指南

## 🎯 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone [项目地址]
cd Meituan

# 创建虚拟环境
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件:

```bash
# LLM配置
LLM_PROVIDER=deepseek  # 可选: deepseek, qwen, openai
DEEPSEEK_API_KEY=sk-xxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com

# 或使用Qwen
QWEN_API_KEY=sk-xxxxx
QWEN_BASE_URL=https://dashscope.aliyuncs.com/api/v1

# 或使用OpenAI
OPENAI_API_KEY=sk-xxxxx
OPENAI_BASE_URL=https://api.openai.com/v1
```

### 3. 数据初始化

```bash
# 1. 加载原始数据到SQLite
python scripts/load_data.py

# 2. 特征工程与数据预处理
python scripts/preprocess.py

# 3. 加载图数据 (可选，用于Node2Vec)
python scripts/load_graph.py
```

### 4. 启动服务

**方式一: 使用启动脚本 (推荐)**

```powershell
# 启动V2版本 (完整功能)
.\start_v2.ps1

# 或启动V1版本 (基础功能)
.\start.ps1
```

**方式二: 手动启动**

```bash
# 后端
cd backend
python app_v2.py

# 前端 (新开终端)
cd frontend
python -m http.server 8080

# 浏览器访问: http://localhost:8080/index_v2.html
```

---

## 📚 功能使用指南

### 1. 智能推荐

**基础推荐 (协同过滤)**

```python
import requests

response = requests.post("http://localhost:8000/api/recommend", json={
    "user_id": "1",
    "top_k": 10,
    "use_llm": False
})

print(response.json())
```

**LLM增强推荐**

```python
response = requests.post("http://localhost:8000/api/recommend", json={
    "user_id": "1",
    "top_k": 5,
    "use_llm": True  # 启用LLM生成推荐理由
})

# 输出包含推荐理由
for rec in response.json()["recommendations"]:
    print(f"{rec['name']}: {rec.get('llm_reason', '无理由')}")
```

### 2. RAG语义搜索

**前端使用**

1. 打开 http://localhost:8080/index_v2.html
2. 在"RAG语义搜索"模块输入查询
3. 查看意图解析结果和搜索结果

**API调用**

```python
response = requests.post("http://localhost:8000/api/rag/search", json={
    "query": "推荐附近评分高的川菜馆",
    "top_k": 10
})

data = response.json()
print("意图:", data["intent"])
print("结果数:", data["total"])
for result in data["results"]:
    print(f"- {result['name']} (评分: {result['score']})")
```

**构建向量索引 (首次使用)**

```python
# 构建POI和SPU向量索引
response = requests.post("http://localhost:8000/api/vector/build")
print(response.json())  # 显示索引统计
```

### 3. Agent运营分析

**获取商家运营报告**

```python
response = requests.post("http://localhost:8000/api/operation/analysis", json={
    "poi_id": "12345"
})

report = response.json()["report"]
print("商家名称:", report["基本信息"]["商家名称"])
print("月均销售额:", report["经营数据"]["月均销售额"])
print("运营建议:")
for category, suggestions in report["运营建议"].items():
    print(f"  {category}:")
    for s in suggestions:
        print(f"    - {s}")
```

**报告结构**

```json
{
  "基本信息": {
    "商家名称": "海底捞火锅",
    "评分": 4.7,
    "分类": "火锅",
    "地址": "朝阳区xxx"
  },
  "经营数据": {
    "总订单量": 15234,
    "月均销售额": 892345,
    "客单价": 128,
    "活跃用户数": 8923
  },
  "热门菜品": [
    {"名称": "招牌毛肚", "销量": 3245, "价格": 38},
    ...
  ],
  "用户画像": {
    "主要年龄段": "25-35岁",
    "男女比例": "45:55",
    "消费偏好": ["火锅", "烧烤"]
  },
  "竞品分析": {
    "周边竞品数": 8,
    "评分排名": "第2名",
    "价格排名": "中等偏上"
  },
  "运营建议": {
    "菜品优化": ["增加低价套餐", "推出季节特色菜"],
    "营销策略": ["工作日午间促销", "会员积分活动"],
    "服务改进": ["缩短等待时间", "优化配送路线"]
  }
}
```

### 4. 评论智能摘要

```python
response = requests.post("http://localhost:8000/api/comment/summary", json={
    "poi_id": "12345"
})

data = response.json()
print("平均评分:", data["average_score"])
print("情感分布:")
print(f"  正面: {data['sentiment']['positive']}%")
print(f"  负面: {data['sentiment']['negative']}%")
print("\n优点:", ", ".join(data["positive_aspects"]))
print("缺点:", ", ".join(data["negative_aspects"]))
print("\nLLM摘要:", data["llm_summary"])
```

### 5. 智能问答

**通用问答**

```python
response = requests.post("http://localhost:8000/api/qa/answer", json={
    "question": "如何提升商家评分？"
})

print(response.json()["answer"])
```

**商家相关问答 (RAG检索)**

```python
response = requests.post("http://localhost:8000/api/qa/answer", json={
    "question": "12345这家店的菜品怎么样？",
    "poi_id": "12345"
})

print(response.json()["answer"])
```

---

## 🔧 高级配置

### 缓存系统

**LLM缓存**

```python
from backend.utils.cache import llm_cache

# 查看缓存统计
stats = llm_cache.get_stats()
print(f"命中率: {stats['hit_rate']:.2%}")
print(f"命中次数: {stats['hits']}")
print(f"未命中次数: {stats['misses']}")

# 清空缓存
llm_cache.clear()
```

**持久化缓存**

```python
from backend.utils.cache import PersistentCache

cache = PersistentCache("my_cache", cache_dir="data/cache")
cache.set("key", {"data": "value"}, ttl=3600)
data = cache.get("key")
```

### 性能监控

**查看API性能指标**

```python
response = requests.get("http://localhost:8000/api/metrics")
metrics = response.json()

print("API调用统计:")
for operation, stats in metrics["api_metrics"].items():
    print(f"  {operation}: 平均{stats['avg']:.3f}s (最快{stats['min']:.3f}s)")

print("\nLLM缓存:")
print(f"  命中率: {metrics['cache']['llm_cache']['hit_rate']:.2%}")
```

**查看日志**

```bash
# 查看API日志
tail -f logs/api_v2_20240115.log

# 查看错误日志
tail -f logs/api_v2_error_20240115.log
```

---

## 🧪 测试与调试

### 运行API测试

```bash
# 完整测试所有端点
python test_api.py

# 测试特定功能
python -c "
import requests
resp = requests.post('http://localhost:8000/api/recommend', json={'user_id': '1'})
print(resp.json())
"
```

### 常见问题排查

**问题1: 数据库连接失败**

```bash
# 检查数据库文件是否存在
ls data/db/meituan.db

# 重新初始化数据库
python scripts/load_data.py
```

**问题2: LLM调用失败**

```bash
# 检查环境变量
echo $DEEPSEEK_API_KEY

# 测试API连接
curl -X POST https://api.deepseek.com/v1/chat/completions \
  -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
  -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"测试"}]}'
```

**问题3: 向量索引未构建**

```bash
# 手动构建索引
curl -X POST http://localhost:8000/api/vector/build

# 或在前端点击"构建向量索引"按钮
```

---

## 📊 性能优化建议

### 1. 缓存策略

```python
# 推荐使用持久化缓存存储常用数据
from backend.utils.cache import PersistentCache

poi_cache = PersistentCache("poi_info", ttl=86400)  # 24小时

def get_poi_info(poi_id):
    cached = poi_cache.get(poi_id)
    if cached:
        return cached
    
    # 从数据库查询
    info = query_database(poi_id)
    poi_cache.set(poi_id, info)
    return info
```

### 2. 批量处理

```python
from backend.utils.cache import BatchProcessor

processor = BatchProcessor(max_batch_size=50, timeout=1.0)

# 批量获取推荐
def batch_recommend(user_ids):
    def process_batch(batch):
        return [get_recommendations(uid) for uid in batch]
    
    return processor.process(user_ids, process_batch)
```

### 3. 异步处理

```python
# 使用FastAPI的异步特性
@app.post("/api/recommend")
async def recommend(request: RecommendRequest):
    # 并发查询数据库和LLM
    import asyncio
    
    db_task = asyncio.to_thread(query_database, request.user_id)
    llm_task = asyncio.to_thread(llm_generate, prompt) if request.use_llm else None
    
    db_result = await db_task
    llm_result = await llm_task if llm_task else None
    
    return merge_results(db_result, llm_result)
```

---

## 🎨 前端自定义

### 修改样式

编辑 `frontend/static/css/style_v2.css`:

```css
/* 修改主题色 */
:root {
    --primary-color: #ff6600;  /* 橙色 → 自定义颜色 */
    --secondary-color: #333333;
}

/* 修改卡片样式 */
.module-card {
    border-radius: 12px;  /* 圆角 */
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
```

### 添加新模块

1. 在 `index_v2.html` 添加HTML结构
2. 在 `app_v2.js` 添加Vue方法
3. 在 `app_v2.py` 添加API端点

```html
<!-- index_v2.html -->
<div class="module-card">
    <h3>新功能模块</h3>
    <button @click="newFeature()">执行</button>
    <div>{{ newResult }}</div>
</div>
```

```javascript
// app_v2.js
async newFeature() {
    const response = await axios.post('/api/new-feature', {...});
    this.newResult = response.data;
}
```

```python
# app_v2.py
@app.post("/api/new-feature")
async def new_feature(request: Request):
    return {"result": "success"}
```

---

## 📦 部署指南

### Docker部署

```dockerfile
# Dockerfile
FROM python:3.9

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "backend/app_v2.py"]
```

```bash
# 构建镜像
docker build -t meituan-platform .

# 运行容器
docker run -p 8000:8000 -v $(pwd)/data:/app/data meituan-platform
```

### 生产环境配置

```bash
# 使用Gunicorn
pip install gunicorn
gunicorn backend.app_v2:app --workers 4 --bind 0.0.0.0:8000

# 使用Nginx反向代理
# /etc/nginx/sites-available/meituan
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
    }
}
```

---

## 🔐 安全建议

1. **API密钥保护**: 永远不要将 `.env` 文件提交到Git
2. **速率限制**: 生产环境添加API限流
3. **输入验证**: 所有用户输入都应验证
4. **HTTPS**: 生产环境使用SSL证书

---

## 📞 支持与反馈

- 问题报告: [GitHub Issues]
- 技术文档: [Wiki]
- 邮箱联系: [your-email]

---

**祝使用愉快！** 🎉
