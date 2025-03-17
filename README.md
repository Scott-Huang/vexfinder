# 向量数据库索引参数推荐工具 (VecIndexFinder)

这是一个向量数据库索引参数推荐工具，帮助用户为向量数据库表选择最优的索引参数配置。该工具支持从数据库表中采样数据，测试不同索引参数的性能，并推荐最佳参数组合。

## 主要功能

- 数据采样：从向量数据库表中采样一定比例的数据
- 索引测试：支持多种索引类型（IVF-FLAT, IVF-PQ, HNSW, DiskANN等）
- 参数优化：测试不同参数组合的性能（QPS和召回率）
- 性能分析：基于测试结果推荐最优参数配置
- 结果可视化：生成性能对比图表


## 安装

### 环境要求

- Python 3.10+
- 支持的向量数据库客户端

### 安装

```bash
# 克隆仓库
cd VecIndexFinder

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac


# 安装依赖
pip install -r vecindex_finder/requirements.txt
```

## 使用方法

使用Python模块方式运行：

```bash
python -m vecindex_finder.index_finder --config config.yml
```


### 完整参数说明

```
用法: vecindexfinder [选项]

选项:
  --config FILE                  配置文件路径，默认为config.yml
  --output-dir DIR               输出目录路径，默认为./results
  --reports-dir DIR              报告输出目录路径，默认为./reports
  --skip-sampling                跳过数据采样步骤，直接使用已有的采样数据
  --skip-distance-computation    跳过距离计算步骤，直接使用已有的距离数据
  --parallel-workers INT         并行工作线程数，不指定则使用配置文件中的设置
  --only-report                  只根据已有的分析结果生成报告，不进行分析
  --result-dir DIR               如果只生成报告，需指定结果目录路径
```


### 示例

使用自定义配置文件:

```bash
vecindexfinder --config my_config.yml
```

