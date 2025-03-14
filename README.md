# 向量数据库索引参数推荐工具 (VecIndexFinder)

这是一个向量数据库索引参数推荐工具，帮助用户为向量数据库表选择最优的索引参数配置。该工具支持从数据库表中采样数据，测试不同索引参数的性能，并推荐最佳参数组合。

## 主要功能

- 数据采样：从向量数据库表中采样一定比例的数据
- 索引测试：支持多种索引类型（IVF-FLAT, IVF-PQ, HNSW, DiskANN等）
- 参数优化：测试不同参数组合的性能（QPS和召回率）
- 性能分析：基于测试结果推荐最优参数配置
- 结果可视化：生成性能对比图表

## 支持的数据库

- Milvus
- OpenGauss
- PGVector
- 可扩展支持更多数据库

## 安装

### 环境要求

- Python 3.8+
- 支持的向量数据库客户端

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/your-repo/VecIndexFinder.git
cd VecIndexFinder

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
python index_finder.py --db milvus --table my_vectors --indices ivfflat,hnsw
```

### 完整参数说明

```
用法: python index_finder.py [选项]

选项:
  --db TYPE                      向量数据库类型 (milvus, opengauss, pgvector)
  --table NAME                   数据表名称
  --dim INT                      向量维度
  --metric TYPE                  距离度量类型 (l2, ip, cosine)
  --indices TYPES                要测试的索引类型列表，逗号分隔 (ivfflat,hnsw,ivfpq,diskann)
  --sample-ratio FLOAT           索引数据采样比例 (0-1)
  --sample-method METHOD         采样方法 (random, stratified, cluster)
  --query-json FILE              查询数据JSON文件路径
  --query-count INT              自动采样的查询数量
  --topk INT                     查询返回的结果数
  --concurrency INT              并发查询数
  --output DIR                   输出目录
  --config FILE                  配置文件路径
  --optimize-for METRIC          优化目标 (qps, recall, combined)
  --weight-qps FLOAT             QPS权重 (0-1)
  --weight-recall FLOAT          召回率权重 (0-1)
  --verbose                      详细输出模式
```

### TODO
1、根据召回差值，来修改参数的步长

### 示例

1. 基本用法（使用默认配置）:

```bash
python index_finder.py --db milvus --table my_vectors --indices ivfflat,hnsw
```

2. 指定采样比例和查询数据:

```bash
python index_finder.py --db milvus --table my_vectors --indices ivfflat,hnsw --sample-ratio 0.1 --query-count 5000
```

3. 使用自定义配置文件:

```bash
python index_finder.py --db milvus --table my_vectors --config my_config.yml
```

4. 优化QPS（速度优先）:

```bash
python index_finder.py --db milvus --table my_vectors --optimize-for qps --weight-qps 0.8 --weight-recall 0.2
```

5. 优化召回率（准确度优先）:

```bash
python index_finder.py --db milvus --table my_vectors --optimize-for recall --weight-qps 0.2 --weight-recall 0.8
```

