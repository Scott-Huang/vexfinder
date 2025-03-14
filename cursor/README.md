# 向量索引参数分析工具

## 项目概述

向量索引参数分析工具是一个用于优化向量数据库索引查询参数的工具集。它可以自动测试不同的索引参数组合，找到在满足最低召回率要求的前提下性能最佳的参数配置。

## 主要组件

1. **Analyzer（分析器）**：负责测试索引参数和查询参数，找到最佳查询参数
2. **ResultCollector（结果收集器）**：负责收集、存储和分析测试结果
3. **批量分析脚本**：用于自动化测试多种索引参数配置

## 支持的索引类型

- **IVF Flat**：基于倒排文件的平面索引
- **IVF PQ**：基于倒排文件的乘积量化索引
- **HNSW**：层次可导航小世界图索引
- **DiskANN**：磁盘辅助的近似近邻搜索索引

## 核心功能

- **自动参数优化**：根据目标召回率自动调整查询参数
- **批量测试**：支持批量测试多种索引参数配置
- **结果分析**：分析测试结果，找出最佳参数配置
- **性能评估**：评估不同参数配置的查询性能（QPS、召回率等）

## 使用方法

### 单个参数测试

```python
from vecindex_finder.core.analyzer import Analyzer
from vecindex_finder.core.config import IndexConfig, PerformanceConfig
from vecindex_finder.core.types import IndexAndQueryParam

# 创建配置
index_config = IndexConfig(find_index_type="ivfflat", auto=True)
performance = PerformanceConfig(limit=100, min_recall=0.8, tolerance=0.05)

# 创建分析器
analyzer = Analyzer(index_config, db_engine, query_data, performance)

# 定义索引参数
param = IndexAndQueryParam(
    index_type="ivfflat",
    index_param={"nlist": 100, "metric": "l2"},
    query_param={"ivf_probes": 10}
)

# 运行分析
best_param, best_performance = analyzer.analyze(param)
```

### 批量参数测试

使用批量分析脚本：

```bash
python cursor/batch_analyze.py --index-type ivfflat
```

或者：

```bash
python cursor/batch_analyze.py --index-type hnsw
```

## 结果存储

测试结果将存储在 `cursor/analysis_results` 目录下，采用以下结构：

```
cursor/
└── analysis_results/
    └── ivfflat_2023_03_13_19_09_04/
        ├── nlist_100_metric_l2_ivf_probes_15.json
        ├── nlist_200_metric_l2_ivf_probes_10.json
        └── nlist_400_metric_l2_ivf_probes_5.json
```

每个JSON文件包含详细的测试结果，包括索引参数、查询参数、性能指标等。

## 相关文档

- [分析器使用指南](analyzer_readme.md)
- [结果收集器使用指南](result_collector_usage.md)

## 注意事项

1. 运行测试前，请确保向量数据库服务已启动
2. 大规模测试可能需要较长时间，请耐心等待
3. 对于生产环境，建议先在测试环境中验证最佳参数 