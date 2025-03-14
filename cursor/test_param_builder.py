#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试IndexParamBuilder的新添加方法
"""

import sys
import os
import json

# 将项目根目录添加到sys.path
# 确保能够导入vecindex_finder模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vecindex_finder.core.param_builder import IndexParamBuilder

def print_json(data):
    """美化打印JSON数据"""
    print(json.dumps(data, indent=2, ensure_ascii=False))

def test_theoretical_params():
    """测试获取理论参数"""
    # 测试不同数据规模和维度下的理论参数
    test_cases = [
        {"name": "小数据集，低维", "sample_count": 5000, "dimension": 32, "index_type": "ivfflat"},
        {"name": "中等数据集，中等维度", "sample_count": 50000, "dimension": 128, "index_type": "ivfpq"},
        {"name": "大数据集，高维", "sample_count": 1000000, "dimension": 512, "index_type": "hnsw"},
        {"name": "特定索引类型HNSW", "sample_count": 50000, "dimension": 128, "index_type": "hnsw"}
    ]
    
    print("\n=== 理论参数测试 ===")
    
    for case in test_cases:
        print(f"\n## 测试场景: {case['name']}")
        sample_count = case["sample_count"]
        dimension = case["dimension"]
        index_type = case["index_type"]
        
        builder = IndexParamBuilder(sample_count, dimension, index_type)
        theoretical_params = builder.get_theoretical_params()
        
        print(f"样本数量: {sample_count:,}, 维度: {dimension}, 索引类型: {index_type}")
        print("索引参数:")
        print_json(theoretical_params['index_params'])
        print("查询参数:")
        print_json(theoretical_params['query_params'])
    
    # 测试获取参数对
    print("\n## 直接获取指定索引类型的参数对:")
    builder = IndexParamBuilder(50000, 128, "ivfflat")
    index_params, query_params = builder.get_params_pair()
    print("索引参数:")
    print_json(index_params)
    print("查询参数:")
    print_json(query_params)
    
    # 测试临时指定不同的索引类型
    print("\n## 临时指定不同的索引类型:")
    index_params, query_params = builder.get_params_pair("hnsw")
    print("HNSW索引参数:")
    print_json(index_params)
    print("HNSW查询参数:")
    print_json(query_params)

def test_recommend_params():
    """测试参数推荐"""
    # 测试不同数据规模和维度下的参数推荐
    sample_count = 50000
    dimension = 128
    
    print("\n=== 参数推荐测试 ===")
    
    # 测试IVF-FLAT参数推荐
    builder = IndexParamBuilder(sample_count, dimension, "ivfflat")
    current_params = {
        'ivf_nlist': 100,
        'ivf_probes': 10
    }
    
    print("\n## IVF-FLAT参数推荐 (维度:{}, 样本数:{:,}):".format(dimension, sample_count))
    print("当前参数:")
    print_json(current_params)
    
    # 向左推荐（注重性能）
    left_params = builder.recommend_new_params(current_params, 'left')
    print("\n向左推荐（注重性能）:")
    print_json(left_params)
    
    # 向右推荐（注重精度）
    right_params = builder.recommend_new_params(current_params, 'right')
    print("\n向右推荐（注重精度）:")
    print_json(right_params)
    
    # 测试HNSW参数推荐
    builder = IndexParamBuilder(sample_count, dimension, "hnsw")
    current_params = {
        'index_type': 'hnsw',
        'M': 16,
        'efConstruction': 64,
        'ef': 80
    }
    
    print("\n## HNSW参数推荐 (维度:{}, 样本数:{:,}):".format(dimension, sample_count))
    print("当前参数:")
    print_json(current_params)
    
    # 向左推荐（注重性能）
    left_params = builder.recommend_new_params(current_params, 'left')
    print("\n向左推荐（注重性能）:")
    print_json(left_params)
    
    # 向右推荐（注重精度）
    right_params = builder.recommend_new_params(current_params, 'right')
    print("\n向右推荐（注重精度）:")
    print_json(right_params)
    
    # 测试不同数据规模下的参数推荐
    print("\n## 不同数据规模下的HNSW参数推荐:")
    
    # 大数据集
    large_builder = IndexParamBuilder(1000000, dimension, "hnsw")
    large_params = large_builder.recommend_new_params(current_params, 'right')
    print("\n大数据集 (1,000,000) 向右推荐:")
    print_json(large_params)
    
    # 小数据集
    small_builder = IndexParamBuilder(5000, dimension, "hnsw")
    small_params = small_builder.recommend_new_params(current_params, 'right')
    print("\n小数据集 (5,000) 向右推荐:")
    print_json(small_params)

def test_different_dimensions():
    """测试不同维度对参数推荐的影响"""
    sample_count = 50000
    
    print("\n=== 不同维度下的参数推荐测试 ===")
    
    # 低维度
    low_dim = 32
    low_builder = IndexParamBuilder(sample_count, low_dim, "ivfpq")
    current_params = {
        'index_type': 'ivfpq',
        'nlist': 100,
        'nprobe': 10,
        'm': 8,
        'refine_factor': 1
    }
    
    print(f"\n## 低维度 ({low_dim}) IVF-PQ参数推荐:")
    print("当前参数:")
    print_json(current_params)
    
    low_params = low_builder.recommend_new_params(current_params, 'right')
    print("\n向右推荐（注重精度）:")
    print_json(low_params)
    
    # 高维度
    high_dim = 512
    high_builder = IndexParamBuilder(sample_count, high_dim, "ivfpq")
    
    print(f"\n## 高维度 ({high_dim}) IVF-PQ参数推荐:")
    print("当前参数:")
    print_json(current_params)
    
    high_params = high_builder.recommend_new_params(current_params, 'right')
    print("\n向右推荐（注重精度）:")
    print_json(high_params)

if __name__ == "__main__":
    test_theoretical_params()
    test_recommend_params()
    test_different_dimensions() 