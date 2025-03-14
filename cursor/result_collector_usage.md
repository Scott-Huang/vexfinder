# 向量索引结果收集器使用指南

## 功能概述

`ResultCollector` 类是一个结果收集和分析工具，用于存储和分析向量索引的测试结果。它可以与 `Analyzer` 类配合使用，将分析器找到的最佳查询参数保存为JSON文件，并进行数据分析和报告生成。

## 关键功能

1. **存储最佳实践结果**：将分析器找到的最佳查询参数保存到特定格式的文件夹中
2. **结果分析**：找出满足最低召回率要求的最佳参数
3. **结果导出**：将测试结果导出为CSV和JSON文件
4. **生成摘要报告**：生成测试结果的摘要报告

## 存储最佳实践结果使用示例

### 单个结果存储

```python
from core.analyzer import Analyzer
from core.config import Config, IndexConfig, PerformanceConfig
from core.result_collector import ResultCollector
from core.types import IndexAndQueryParam

# 创建配置
config = Config()
index_config = IndexConfig(find_index_type="ivfflat", auto=True)
performance = PerformanceConfig(limit=100, min_recall=0.8, tolerance=0.05)

# 创建分析器和结果收集器
analyzer = Analyzer(index_config, db_engine, query_data, performance)
collector = ResultCollector(config)

# 定义索引参数
param = IndexAndQueryParam(
    index_type="ivfflat",
    index_param={"nlist": 100, "metric": "l2"},
    query_param={"n_probe": 10}
)

# 运行分析
best_param, best_performance = analyzer.analyze(param)

# 存储最佳实践结果
collector.store_best_practice(
    index_type="ivfflat",
    index_param=param.index_param,
    best_param=best_param,
    performance=best_performance
)
```

### 批量结果存储

如果您有多个索引参数配置需要测试和存储结果，可以使用批量存储方法：

```python
# 创建多个索引参数配置
index_params = [
    {"nlist": 100, "metric": "l2"},
    {"nlist": 200, "metric": "l2"},
    {"nlist": 400, "metric": "l2"}
]

# 存储结果列表
results = []

# 分析每个索引参数配置
for idx_param in index_params:
    param = IndexAndQueryParam(
        index_type="ivfflat",
        index_param=idx_param,
        query_param={"n_probe": 10}  # 初始查询参数
    )
    
    # 运行分析
    best_param, best_performance = analyzer.analyze(param)
    
    # 添加到结果列表
    results.append((idx_param, best_param, best_performance))

# 批量存储最佳实践结果
collector.store_best_practices(results)
```

## 存储文件结构

最佳实践结果将存储在 `cursor/analysis_results` 目录下，采用以下结构：

```
cursor/
└── analysis_results/
    └── ivfflat_2023_03_13_19_09_04/
        ├── nlist_100_metric_l2_n_probe_15.json
        ├── nlist_200_metric_l2_n_probe_10.json
        └── nlist_400_metric_l2_n_probe_5.json
```

每个JSON文件包含以下信息：
- 索引类型
- 索引参数
- 最佳查询参数
- 性能指标（召回率、查询时间、QPS等）
- 时间戳

## 结果分析

使用 `ResultCollector` 进行结果分析：

```python
# 假设我们已经添加了多个测试结果
collector.add_results(test_results)

# 找出最佳索引参数
best_params = collector.find_best_params()

# 按索引类型找出最佳参数
best_params_by_type = collector.find_best_params_by_type()

# 生成摘要
summary = collector.generate_summary()

# 导出结果和摘要
collector.export_results("results/test_results.csv")
collector.export_summary("results/summary.json")
```

## 注意事项

1. 存储路径格式为 `{索引类型}_{年}_{月}_{日}_{时}_{分}_{秒}`
2. 文件名格式为索引参数和查询参数的组合，以下划线连接
3. 对于同一类型的索引，不同参数配置的最佳实践会存储在同一个文件夹中
4. 所有结果均以JSON格式存储，便于后续分析和可视化 