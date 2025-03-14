# 向量索引参数优化工具 (VecIndex Finder)

这是一个用于向量数据库创建索引时，查找最佳索引参数的工具。该工具基于索引创建时间、QPS和召回率来评估不同索引参数的性能，并找出最佳配置。

## 功能特点

- 支持多种向量索引类型：IVF-FLAT、IVF-PQ、HNSW
- 自动采样数据和查询向量
- 自动测试不同索引参数组合
- 基于QPS和召回率评估索引性能
- 生成详细的性能报告和可视化图表
- 支持自定义配置参数

## 安装依赖

```bash
pip install psycopg2-binary pandas matplotlib seaborn pyyaml tqdm
```

## 使用方法

1. 配置 `config.yml` 文件，设置数据库连接信息、表信息、采样配置和索引参数等
2. 运行索引参数优化工具

```bash
python vecindex_finder/index_finder.py --config vecindex_finder/config.yml
```

### 命令行参数

- `--config`: 配置文件路径，默认为 `config.yml`
- `--output-dir`: 输出目录路径，默认为 `./cursor/index_finder_results`
- `--skip-sampling`: 跳过数据采样步骤，直接使用已有的采样数据
- `--skip-query-sampling`: 跳过查询采样步骤，直接使用已有的查询数据
- `--skip-distance-computation`: 跳过距离计算步骤，直接使用已有的距离数据

## 配置文件说明

配置文件采用YAML格式，包含以下主要部分：

### 数据库连接配置

```yaml
connection:
  host: "127.0.0.1"
  port: 6432
  user: "ann"
  password: "Huawei@123"
  dbname: "ann"
```

### 表信息配置

```yaml
table_info:
  # 表名
  table_name: "items"
  # 向量字段名
  vector_column_name: "embedding"
  # 向量维度
  dimension: 128
  # 距离度量类型, l2/ip/cosine
  metric: "l2"
```

### 采样配置

```yaml
sampling:
  # 默认采样比例
  default_ratio: 0.1
  # 最小采样数量
  min_sample_count: 10000
  # 最大采样数量
  max_sample_count: 1000000
```

### 查询配置

```yaml
query:
  # query获取方式, sample/从原表中采样, json/从json文件中获取
  query_get_type: "sample"
  # 测试的请求数量，默认1000，query_get_type为sample时有效
  query_count: 1000
  # query数据路径, 如果为空，则使用默认的采样方式
  query_data_path: "query_data.json"
```

### 性能测试配置

```yaml
performance:
  # limit, 每次查询的限制数量
  limit: 100
  # 最低容忍召回率
  min_recall: 0.8
```

### 索引配置

```yaml
index:
  # 需要查找的索引类型
  find_index_type: "ivfflat"
  # 是否自动选择最佳参数
  auto: true
  # IVF-FLAT索引参数
  ivfflat:
    # 聚类中心数量
    nlist: [16, 32, 64, 128, 256, 512, 1024]
    # 查询参数
    nprobe: [1, 4, 8, 16, 32, 64, 128]

  # IVF-PQ索引参数
  ivfpq:
    # 聚类中心数量
    nlist: [16, 32, 64, 128, 256, 512, 1024]
    # 子量化器数量
    m: [4, 8, 16, 32]
    # 每个子量化器的bit数
    nbits: 8
    # 查询参数
    nprobe: [1, 4, 8, 16, 32, 64]
    # 精炼因子范围
    ivfpq_refine_k_factor: [1, 2, 4, 8]
  
  # HNSW索引参数
  hnsw:
    # 每个节点的最大连接数
    M: [8, 16, 32, 64]
    # 构建时的候选邻居数量
    efConstruction: [40, 80, 120, 200, 400]
    # 查询时的候选邻居数量
    ef: [10, 20, 40, 80, 100, 200, 400, 800]
```

## 输出结果

工具会在指定的输出目录生成以下内容：

1. CSV和JSON格式的测试结果文件
2. 性能报告目录，包含：
   - QPS vs 召回率散点图
   - 索引创建时间 vs QPS散点图
   - 各参数对性能影响的箱线图
   - 最佳参数报告（CSV和HTML格式）

## 示例输出

运行工具后，会在控制台输出最佳索引参数，例如：

```
==================================================
最佳索引参数:
索引类型: ivfflat
nlist: 256
nprobe: 32
QPS: 1254.32
平均召回率: 0.9876
索引创建时间: 12.34 秒
==================================================

详细报告已生成到目录: ./cursor/index_finder_results/report_20230101_123456
```

## 注意事项

1. 确保数据库中存在配置的表和向量列
2. 测试过程可能需要较长时间，特别是当参数组合较多时
3. 可以使用命令行参数跳过某些步骤，加快测试速度
4. 输出目录会自动创建，无需手动创建

## 开发者

如需扩展或修改工具功能，可以参考以下文件：

- `core/analyzer.py`: 索引分析器，负责测试不同索引参数的性能
- `core/visualizer.py`: 可视化工具，负责生成性能报告和图表
- `core/config.py`: 配置管理类，负责加载和验证配置
- `core/engine.py`: 数据库引擎，负责管理数据库连接
- `sampling.py`: 采样器，负责数据和查询采样
- `index_finder.py`: 主程序，负责协调各组件工作 