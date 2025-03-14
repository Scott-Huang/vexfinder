# IndexParamBuilder新增功能说明文档

## 概述

本文档描述了`IndexParamBuilder`类新增的两个主要功能：

1. **获取理论参数** - 根据数据集维度和行数，提供理论上最优的索引参数和查询参数
2. **参数推荐** - 根据用户提供的当前参数，向性能或精度方向推荐新的参数

这些功能旨在帮助用户更快地找到适合其场景的最佳索引参数，减少反复测试的时间。

## 类初始化与基本结构

`IndexParamBuilder`类的构造函数接收以下参数：

```python
def __init__(self, sample_table_count: int, dimension: int, index_type: str):
    """
    初始化索引参数构建器
    
    Args:
        sample_table_count: 样本数量
        dimension: 数据维度
        index_type: 索引类型，如'ivfflat', 'ivfpq', 'hnsw'等
    """
```

- `sample_table_count`: 数据表的样本数量
- `dimension`: 向量维度
- `index_type`: 索引类型（例如'ivfflat', 'ivfpq', 'hnsw'等）

## 获取理论参数

### 功能描述

理论参数是指根据向量数据的维度、数据量等特征，计算出的理论上最优的索引参数。这些参数会根据不同的数据规模和维度进行智能调整，为用户提供最佳的起点参数。

### 相关方法

#### `get_theoretical_params()`

返回指定索引类型的理论最优参数，包括索引构建参数和查询参数。索引类型通过初始化时提供的`index_type`确定。如果未指定索引类型，则默认使用'ivfflat'。

**返回格式**：
```python
{
    'index_params': {...},  # 索引构建参数
    'query_params': {...}   # 查询参数
}
```

#### `get_params_pair(index_type=None)`

获取指定索引类型的理论最优索引参数和查询参数对。

**参数**：
- `index_type`: 索引类型，如'ivfflat', 'ivfpq', 'hnsw'等。如果为None，则使用实例的索引类型。

**返回值**：
- 返回一个元组 `(index_params, query_params)`，包含索引参数和查询参数

**说明**：
这个方法会暂时修改实例的`index_type`（如果提供了不同的索引类型），调用`get_theoretical_params()`获取参数，然后恢复原始的`index_type`。这样用户可以在不改变实例状态的情况下获取不同索引类型的参数。

### 工作原理

对于每种索引类型，我们基于以下原则计算理论参数，并根据数据规模和维度进行调整：

1. **IVF-FLAT**：
   - `nlist` = 样本数量的平方根，对小数据集和大数据集会进行特殊调整
     - 小数据集(<10K)：使用较小的nlist值(16-64)
     - 大数据集(>1M)：使用较大的nlist值(nlist*1.2)
   - `nprobe` = `nlist` * 比例因子，比例因子会根据数据量调整：
     - 小数据集: 10%
     - 中等数据集: 5%
     - 大数据集: 3%

2. **IVF-PQ**：
   - `nlist` = 同IVF-FLAT
   - `m` = 针对不同维度优化，低维使用8，高维使用维度的1/8(不超过32)，且优先选择维度的因子
   - `nbits` = 8（固定值）
   - `nprobe` = 根据数据量调整比例
   - `refine_factor` = 低维使用1，高维(>128)使用2

3. **HNSW**：
   - `M`值根据维度和数据量综合选择：
     - 维度 ≤ 64: 基础M=16
     - 维度 ≤ 256: 基础M=24
     - 维度 > 256: 基础M=32
     - 小数据集: 基础M+4
     - 大数据集: 基础M-4
   - `efConstruction` = M * 数据量相关因子(4-8)
   - `ef` = `efConstruction` * 维度和数据量相关比例(1.5-2.5)

## 参数推荐

### 功能描述

参数推荐功能允许用户根据当前使用的参数，向性能优先或精度优先的方向获取新的推荐参数。新实现会考虑数据维度和样本数量，提供更加精准的推荐。

### 相关方法

#### `recommend_new_params(current_params, direction)`

基于当前参数，向指定方向推荐新的参数。

**参数**：
- `current_params`: 当前使用的索引参数
- `direction`: 推荐方向，'left'表示更注重性能，'right'表示更注重精度，默认为'right'

**返回值**：
- 返回推荐的新参数

### 工作原理

参数推荐基于以下原则，并根据数据维度和样本量智能调整：

1. **向左推荐（性能优先）**：
   - 调整因子会根据数据规模和维度调整，大数据集或高维数据使用更温和的调整因子(0.85)
   
   - 对于IVF系列索引：
     - 增加`nlist`值（但小数据集增加幅度较小）
     - 减少`nprobe`值
     - 对于IVF-PQ，减少`m`值（优先选择维度的因子）和`refine_factor`值
   
   - 对于HNSW索引：
     - 减少`M`值（但确保低维数据≥8，高维数据≥12）
     - 减少`efConstruction`值（但至少为M的2倍）
     - 减少`ef`值（但至少等于M）

2. **向右推荐（精度优先）**：
   - 调整因子会根据数据规模和维度调整，大数据集或高维数据使用更温和的调整因子(1.15)
   
   - 对于IVF系列索引：
     - 减少`nlist`值（但确保至少为4或16，取决于数据量）
     - 增加`nprobe`值（高维数据增幅更大）
     - 对于IVF-PQ，增加`m`值（优先选择维度的因子）和`refine_factor`值（高维数据增幅更大）
   
   - 对于HNSW索引：
     - 增加`M`值（但低维数据最大48，高维数据最大64）
     - 增加`efConstruction`值（大数据集增幅更大）
     - 增加`ef`值（高维数据增幅更大）

## 使用示例

以下是使用这些新功能的简单示例：

```python
from vecindex_finder.core.param_builder import IndexParamBuilder

# 创建参数构建器实例，指定样本数量、维度和索引类型
sample_count = 50000
dimension = 128
builder = IndexParamBuilder(sample_count, dimension, "hnsw")

# 获取理论参数
theoretical_params = builder.get_theoretical_params()
print(f"索引参数: {theoretical_params['index_params']}")
print(f"查询参数: {theoretical_params['query_params']}")

# 直接获取理论参数对
index_params, query_params = builder.get_params_pair()
print(f"索引参数: {index_params}")
print(f"查询参数: {query_params}")

# 获取不同索引类型的参数对，不改变实例的索引类型
ivf_index_params, ivf_query_params = builder.get_params_pair("ivfflat")
print(f"IVF-FLAT索引参数: {ivf_index_params}")

# 根据当前参数推荐新参数
current_params = {
    'index_type': 'hnsw',
    'M': 16,
    'efConstruction': 64,
    'ef': 80
}

# 向性能方向推荐
perf_params = builder.recommend_new_params(current_params, 'left')
print(f"性能优先推荐参数: {perf_params}")

# 向精度方向推荐
prec_params = builder.recommend_new_params(current_params, 'right')
print(f"精度优先推荐参数: {prec_params}")
```

## 改进点总结

相比旧版实现，新版`IndexParamBuilder`类有以下改进：

1. **简化的返回格式**：`get_theoretical_params`现在直接返回单层字典，格式为`{"index_params": {...}, "query_params": {...}}`，使用更加直观
2. **根据索引类型返回参数**：根据初始化时指定的索引类型或默认类型返回对应的参数，无需额外过滤
3. **灵活的参数获取**：`get_params_pair`支持临时指定不同的索引类型，不影响实例的原始配置
4. **考虑维度和数据量**：所有参数计算都充分考虑数据维度和样本数量，提供更精准的参数建议
5. **智能调整因子**：对不同场景使用不同的调整因子，避免过度调整
6. **优先维度因子**：对于PQ索引，优先选择维度的因子作为m值
7. **边界保护**：确保推荐的参数在合理范围内，避免极端值
8. **可定制性**：类初始化时即可指定索引类型，方便用户针对特定索引类型获取参数

## 注意事项

1. 理论参数只是一个起点，实际最优参数可能因数据分布、硬件环境等因素而异
2. 参数推荐是基于当前参数的增量式调整，不是全局最优解
3. 不同维度和数据量的组合可能需要不同的参数策略，建议结合实际性能测试结果来决定最终使用的参数
4. 如果需要频繁切换索引类型，建议使用`get_params_pair`方法的`index_type`参数，而不是创建多个实例

## 进一步优化建议

1. 考虑增加学习功能，根据历史测试结果自动推荐更优参数
2. 加入更详细的数据分布分析，如向量分布的均匀性、聚类特性等
3. 支持更多索引类型和参数组合
4. 增加参数敏感度分析，识别对性能影响最大的参数
5. 提供可视化工具，帮助用户理解参数调整对性能和精度的影响 