# 数据字典

## 概述
本数据字典用于说明平台所使用的数据结构，包括用户信息、商家信息、菜品信息、订单数据等。

**数据来源**：美团外卖推荐数据集（Meituan TRD），包含 2021年3月1日-3月28日 北京11个商圈的订单数据。
- **训练集**：3月1日-3月21日（前三周）
- **测试集**：3月22日-3月28日（最后一周）
- **数据规模**：400,000+ 节点，18M+ 边（图数据）

#### 版本：1.1（修正字段名、补充测试集说明、统一格式）

---

## 表结构说明

### 1. users.txt（用户属性表）

| 字段名 | 类型 | 含义 | 示例 | 备注 |
|--------|------|------|------|------|
| user_id | int | 用户ID | 0 | 非用户真实ID，已脱敏 |
| avg_pay_amt | string | 历史平均消费区间 | [36,49) | 共5个区间：<29, [29,36), [36,49), [49,65), >=65 |
| avg_pay_amt_weekdays | string | 工作日历史平均消费区间 | [29,36) | 同上 |
| avg_pay_amt_weekends | string | 周末历史平均消费区间 | [36,49) | 同上 |

---

### 2. pois.txt（商家信息表）

**注意**：字段顺序中 `primary_second_tag_id` 和 `primary_third_tag_id` 在 `primary_first_tag_id` 之前。

| 字段名 | 类型 | 含义 | 示例 | 备注 |
|--------|------|------|------|------|
| wm_poi_id | int | 商家ID | 0 | 非商家真实ID，已脱敏 |
| wm_poi_name | string | 商家名称 | 302658-52727-042150... | 脱敏后的商家名称 |
| primary_second_tag_id | int | 二级品类ID | 0 | 注意：在文件中位于 first_tag 之前 |
| primary_third_tag_id | int | 三级品类ID | 0 | |
| primary_first_tag_id | int | 一级品类ID | 0 | |
| poi_brand_id | int | 品牌ID | 0 | |
| aor_id | int | 商家所属商圈ID | 0 | 取值范围 0-10，共11个商圈 |
| poi_score | float | 商家综合评分 | 4.66 | |
| delivery_comment_avg_score | float | 配送评分 | 4.87 | |
| food_comment_avg_score | float | 菜品评分 | 4.61 | |

---

### 3. spus.txt（菜品信息表）

| 字段名 | 类型 | 含义 | 示例 | 备注 |
|--------|------|------|------|------|
| wm_food_spu_id | int | 菜品ID | 0 | 非菜品真实ID，已脱敏 |
| wm_food_spu_name | string | 菜品名称 | a22084-de2424-11949... | 脱敏后的菜品名称 |
| price | float | 价格（元） | 32.0 | |
| category | list | 菜品分类标签列表 | [0] 或 [1] | JSON数组格式，可包含多个分类ID |
| ingredients | list | 食材标签列表 | [0] 或 [1,2,3,4,5,6,7] | JSON数组格式，可为 NULL |
| taste | list | 口味标签列表 | [0] 或 [6] | JSON数组格式，可为 NULL |
| stand_food_id | int | 食品标准ID | 0 | 可为 NULL |
| stand_food_name | string | 食品标准名称 | a51005-4a2318 | 脱敏后的标准菜品名称，可为 NULL |

---

### 4. orders_train.txt（训练集订单-餐厅关系表）

| 字段名 | 类型 | 含义 | 示例 | 备注 |
|--------|------|------|------|------|
| user_id | int | 用户ID | 120037 | 外键，关联 users 表 |
| wm_order_id | int | 订单ID | 0 | 主键，从0开始自增 |
| wm_poi_id | int | 商家ID | 1806 | 外键，关联 pois 表 |
| aor_id | int | 商家所属商圈ID | 0 | 取值 0-10 |
| order_price_interval | string | 订单价格区间 | <29 | 共5个区间：<29, [29,36), [36,49), [49,65), >=65 |
| order_timestamp | int | 订单时间戳 | 1614557985 | Unix时间戳（秒） |
| ord_period_name | int | 订单所处时段 | 0 | 0=早餐, 1=午餐, 2=晚餐, 3=夜宵 |
| order_scene_name | string | 下单场景ID | 0 或 未知 | 可能的值：0, 1, 未知（中文字符） |
| aoi_id | int | 用户收餐地址ID | 0 | 可为 NULL |
| takedlvr_aoi_type_name | string | 用户收餐地址类型 | 0 或 未知 | 可能的值：0, 1, 未知（中文字符），可为 NULL |
| dt | string | 订单日期 | 20210301 | 格式：YYYYMMDD，训练集日期范围：20210301-20210321 |

---

### 5. orders_spu_train.txt（训练集订单-菜品关系表）

| 字段名 | 类型 | 含义 | 示例 | 备注 |
|--------|------|------|------|------|
| wm_order_id | int | 订单ID | 43925 | 外键，关联 orders_train 表 |
| wm_food_spu_id | int | 菜品ID | 142475 | 外键，关联 spus 表 |
| dt | string | 订单日期 | 20210301 | 格式：YYYYMMDD |

---

### 6. orders_poi_session.txt（用户点击序列表）

| 字段名 | 类型 | 含义 | 示例 | 备注 |
|--------|------|------|------|------|
| wm_order_id | int | 订单ID | 21559 | 外键，关联 orders_train 表 |
| clicks | string | 用户在下单前点击的餐厅序列 | 1782#535#1002#26705#975 | 数值为餐厅ID，用 # 分隔，表示浏览顺序，可为空 |
| dt | string | 订单日期 | 20210301 | 格式：YYYYMMDD |

---

### 7. orders_test_poi.txt（测试集订单-餐厅特征表）

**说明**：测试集不包含标签字段（wm_poi_id），需要通过模型预测。

| 字段名 | 类型 | 含义 | 示例 | 备注 |
|--------|------|------|------|------|
| user_id | int | 用户ID | 140760 | 外键，关联 users 表 |
| wm_order_id | int | 订单ID | 1068495 | 主键，测试集订单ID从训练集最大值继续递增 |
| aor_id | int | 商家所属商圈ID | 6 | 取值 0-10 |
| order_timestamp | int | 订单时间戳 | 1616379321 | Unix时间戳（秒） |
| ord_period_name | int | 订单所处时段 | 1 | 0=早餐, 1=午餐, 2=晚餐, 3=夜宵 |
| aoi_id | int | 用户收餐地址ID | 169 | 可为 NULL |
| takedlvr_aoi_type_name | string | 用户收餐地址类型 | 1 或 未知 | 可能的值：0, 1, 未知（中文字符），可为 NULL |
| dt | string | 订单日期 | 20210322 | 格式：YYYYMMDD，测试集日期范围：20210322-20210328 |

---

### 8. orders_test_spu.txt（测试集订单-菜品特征表）

**说明**：测试集不包含标签字段（wm_food_spu_id），需要通过模型预测。一个订单可包含多条记录。

| 字段名 | 类型 | 含义 | 示例 | 备注 |
|--------|------|------|------|------|
| user_id | int | 用户ID | 168405 | 外键，关联 users 表 |
| wm_order_id | int | 订单ID | 1073854 | 非唯一，一个订单可对应多个菜品 |
| aor_id | int | 商家所属商圈ID | 5 | 取值 0-10 |
| order_timestamp | int | 订单时间戳 | 1616384386 | Unix时间戳（秒） |
| ord_period_name | int | 订单所处时段 | 1 | 0=早餐, 1=午餐, 2=晚餐, 3=夜宵 |
| aoi_id | int | 用户收餐地址ID | NULL | 可为 NULL |
| takedlvr_aoi_type_name | string | 用户收餐地址类型 | 未知 | 可能的值：0, 1, 未知（中文字符），可为 NULL |
| dt | string | 订单日期 | 20210322 | 格式：YYYYMMDD |

---

### 9. orders_poi_test_label.txt（测试集订单-餐厅标签表）

**说明**：用于评估餐厅推荐模型的真实标签。

| 字段名 | 类型 | 含义 | 示例 | 备注 |
|--------|------|------|------|------|
| user_id | int | 用户ID | 140760 | 外键，关联 users 表 |
| wm_order_id | int | 订单ID | 1068495 | 外键，关联 orders_test_poi 表 |
| wm_poi_id | int | 商家ID（真实标签） | 2198 | 用户实际下单的商家ID |
| dt | string | 订单日期 | 20210322 | 格式：YYYYMMDD |

---

### 10. orders_spu_test_label.txt（测试集订单-菜品标签表）

**说明**：用于评估菜品推荐模型的真实标签。

| 字段名 | 类型 | 含义 | 示例 | 备注 |
|--------|------|------|------|------|
| user_id | int | 用户ID | 168405 | 外键，关联 users 表 |
| wm_order_id | int | 订单ID | 1073854 | 外键，关联 orders_test_spu 表 |
| wm_food_spu_id | int | 菜品ID（真实标签） | 16481 | 用户实际下单的菜品ID |
| dt | string | 订单日期 | 20210322 | 格式：YYYYMMDD |

---

### 11. graph.bin（图结构数据）

**说明**：包含用户、商家（POI）、菜品（SPU）等节点及其关系的图结构数据。

**数据规模**：
- **节点数**：400,000+
- **边数**：18,000,000+

**节点类型**：
- User节点：用户
- POI节点：商家
- SPU节点：菜品

**边类型**：
- User-POI：用户订单关系
- User-SPU：用户购买关系
- POI-SPU：商家提供菜品关系
- User-User、POI-POI等：可能包含协同过滤构建的相似性边

**加载方式**：
```python
import dgl
# 使用 DGL 库加载图数据
graph = dgl.load_graphs('Meituan_TRD/graph.bin')
```

**应用场景**：
- 图神经网络（GNN）模型训练
- 节点嵌入（Node Embedding）
- 图算法（PageRank、社区发现等）
- 多跳关系推理

---

## 数据使用建议

### 文件组织
```
Meituan/
├── data/
│   ├── raw/              # 原始数据（不要修改）
│   │   └── Meituan_TRD/  # 数据集文件夹
│   ├── processed/        # 处理后的数据（CSV、Parquet等）
│   └── db/               # 数据库文件
│       └── meituan.db    # SQLite数据库
├── scripts/              # 数据处理脚本
│   ├── load_data.py      # 数据加载脚本
│   └── preprocess.py     # 数据预处理脚本
└── notebooks/            # Jupyter notebooks
```

### 数据导入流程
1. **解析文本文件**：所有 `.txt` 文件为 Tab 分隔的文本文件（TSV格式）
2. **清洗数据**：处理 NULL 值、未知值（中文字符）
3. **导入数据库**：建议使用 SQLite（开发）或 MySQL（生产）
4. **建立索引**：为外键字段建立索引以提升查询性能
5. **加载图数据**：使用 DGL 或 PyG 加载 graph.bin

### 字段值说明

**价格区间**（5个区间）：
- `<29`：小于29元
- `[29,36)`：29元（含）到36元（不含）
- `[36,49)`：36元（含）到49元（不含）
- `[49,65)`：49元（含）到65元（不含）
- `>=65`：大于等于65元

**订单时段**（4个时段）：
- `0`：早餐
- `1`：午餐
- `2`：晚餐
- `3`：夜宵

**特殊值处理**：
- `NULL`：数据库空值
- `未知`（中文字符）：未知或缺失的分类值
- 空字符串：点击序列可能为空

### 推荐任务定义

1. **餐厅推荐任务**：
   - 输入：用户特征 + 订单上下文（时间、地址等）
   - 输出：预测用户会选择的餐厅（wm_poi_id）
   - 评估：使用 orders_poi_test_label.txt

2. **菜品推荐任务**：
   - 输入：用户特征 + 订单上下文
   - 输出：预测用户会购买的菜品（wm_food_spu_id）
   - 评估：使用 orders_spu_test_label.txt

---

## 参考资料

- 数据集官方说明：`Meituan_TRD/README.md`
- DGL 图加载文档：https://docs.dgl.ai/
- 项目需求文档：`requests.md`

---

## 更新日志

- **v1.1** (2025-12-28)
  - 修正 pois 表字段名（tag_name → tag_id）
  - 修正 spus 表字段名（standfood → stand_food）
  - 补充 orders_train.txt 缺失字段
  - 新增测试集4个表的完整说明
  - 统一所有表格格式
  - 完善 graph.bin 详细说明
  - 修正数据类型标注
  - 添加字段值枚举说明
  
- **v1.0** (初始版本)
  - 基于 Meituan_TRD/README.md 创建
