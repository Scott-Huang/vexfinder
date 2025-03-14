# 向量索引参数分析器

## 功能说明

向量索引参数分析器是一个用于自动寻找最优查询参数的工具。它能够根据用户指定的召回率要求，自动调整查询参数，找到满足要求的最小参数值，从而在保证查询质量的同时提高查询性能。

## 参数调整策略

分析器主要实现了两种参数调整策略：

1. **参数增大策略**：当初始参数得到的召回率低于目标值时，分析器会逐步增大参数值，直到达到目标召回率。

2. **参数减小策略**：当初始参数得到的召回率高于目标值较多时，分析器会逐步减小参数值，直到召回率接近但不低于目标值，以找到满足要求的最小参数值。

## 支持的索引类型及参数

分析器支持以下索引类型的参数调整：

- **IVF Flat**: 调整 `n_probe` 参数
- **IVF PQ**: 调整 `n_probe` 参数
- **HNSW**: 调整 `ef_search` 参数
- **DiskANN**: 调整 `search_list` 参数

## 分析结果

分析器会在 `cursor/analysis_results` 目录下生成分析结果文件，包含以下信息：

- 索引创建时间
- 索引大小
- 表大小
- 索引类型
- 索引参数
- 初始查询参数
- 最佳查询参数
- 召回率
- 平均查询时间
- 最小查询时间
- 最大查询时间
- QPS (每秒查询数)

## 使用方法

```python
from core.analyzer import Analyzer
from core.config import IndexConfig, PerformanceConfig
from core.types import IndexAndQueryParam

# 创建配置
index_config = IndexConfig(find_index_type="hnsw", auto=True, prepare_cache=True)
performance = PerformanceConfig(limit=100, min_recall=0.8)

# 创建分析器
analyzer = Analyzer(index_config, db_engine, query_data, performance)

# 定义初始参数
param = IndexAndQueryParam(
    index_type="hnsw",
    index_param={"m": 16, "efConstruction": 200},
    query_param={"ef_search": 100}
)

# 分析参数
best_param, best_performance = analyzer.analyze(param)
```

## 配置说明

- **min_recall**: 目标召回率，分析器会尝试找到达到此召回率的最小参数值
- **limit**: 查询时返回的最大结果数

## 结果解释

分析器会返回以下结果：

- **best_param**: 找到的最佳查询参数
- **best_performance**: 最佳参数对应的性能指标，包括召回率、查询时间和QPS 