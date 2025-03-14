#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量向量索引参数分析示例脚本
"""

import sys
import os

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vecindex_finder.core.analyzer import Analyzer
from vecindex_finder.core.config import Config, IndexConfig, PerformanceConfig
from vecindex_finder.core.result_collector import ResultCollector
from vecindex_finder.core.types import IndexAndQueryParam
from vecindex_finder.core.engine import db_engine


def run_batch_analysis(index_type="ivfflat", query_data=None):
    """
    运行批量向量索引参数分析
    
    Args:
        index_type: 索引类型
        query_data: 查询数据
    """
    # 如果未提供查询数据，这里需要加载测试数据
    if query_data is None:
        # 这里应该实现加载测试数据的逻辑
        # 为了示例，我们使用一个空列表
        query_data = []
        print("警告: 未提供查询数据，这只是一个演示")
    
    # 创建配置
    config = Config()
    index_config = IndexConfig(find_index_type=index_type, auto=True, prepare_cache=True)
    performance = PerformanceConfig(limit=100, min_recall=0.8, weight={"create_index_time": 0.3, "index_size": 0.1, "qps": 0.6})
    
    # 创建分析器和结果收集器
    analyzer = Analyzer(index_config, db_engine, query_data, performance)
    collector = ResultCollector(config)
    
    # 定义要测试的索引参数配置
    if index_type == "ivfflat":
        # IVF Flat索引参数
        index_params = [
            {"nlist": 100, "metric": "l2", "table_name": "items", "vector_column_name": "embedding"},
            {"nlist": 200, "metric": "l2", "table_name": "items", "vector_column_name": "embedding"},
            {"nlist": 400, "metric": "l2", "table_name": "items", "vector_column_name": "embedding"}
        ]
        initial_query_param = {"ivf_probes": 10}
    elif index_type == "hnsw":
        # HNSW索引参数
        index_params = [
            {"m": 16, "efConstruction": 100, "metric": "l2", "table_name": "items", "vector_column_name": "embedding"},
            {"m": 24, "efConstruction": 200, "metric": "l2", "table_name": "items", "vector_column_name": "embedding"},
            {"m": 32, "efConstruction": 400, "metric": "l2", "table_name": "items", "vector_column_name": "embedding"}
        ]
        initial_query_param = {"hnsw_ef_search": 100}
    else:
        print(f"不支持的索引类型: {index_type}")
        return
    
    # 存储结果列表
    results = []
    
    # 分析每个索引参数配置
    for idx_param in index_params:
        print(f"\n正在分析索引参数: {idx_param}")
        param = IndexAndQueryParam(
            index_type=index_type,
            index_param=idx_param,
            query_param=initial_query_param
        )
        
        try:
            # 运行分析
            best_param, best_performance = analyzer.analyze(param)
            
            # 添加到结果列表
            results.append((idx_param, best_param, best_performance))
            
            print(f"分析完成: 最佳查询参数 = {best_param}")
            print(f"性能: 召回率 = {best_performance['recall']:.4f}, QPS = {best_performance['qps']:.2f}")
        except Exception as e:
            print(f"分析出错: {e}")
    
    # 批量存储最佳实践结果
    if results:
        filepaths = collector.store_best_practices(results)
        print(f"\n已存储 {len(filepaths)} 个最佳实践结果")
        for filepath in filepaths:
            print(f"  - {filepath}")
    else:
        print("\n没有可存储的结果")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="批量向量索引参数分析工具")
    parser.add_argument("--index-type", default="ivfflat", choices=["ivfflat", "hnsw"],
                        help="要分析的索引类型")
    args = parser.parse_args()
    
    # 运行批量分析
    run_batch_analysis(index_type=args.index_type) 