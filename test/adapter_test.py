#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import logging

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vecindex_finder.sampling import Sampling
from vecindex_finder.core.config import Config
from vecindex_finder.core.engine import DatabaseEngine
from vecindex_finder.core.logging import logger


def test_sampling():
    """测试OpenGaussAdapter的基本功能"""
    logger.info("开始测试OpenGaussAdapter")
    
    config = Config()
    db_engine = DatabaseEngine(config)
    sampling = Sampling(config, db_engine)
    # 创建适配器实例
    adapter = None
    try:
        adapter = OpenGaussAdapter()
        logger.info("成功创建OpenGaussAdapter实例")
        
        # 测试采样数据
        sample_table = adapter.sample_data()
        logger.info(f"成功采样数据到表: {sample_table}")
        
        # 测试采样查询数据
        query_table = adapter.sample_query_data()
        logger.info(f"成功采样{adapter._query_count}条查询数据到表: {query_table}")
        
        # 测试计算最近邻
        adapter.compute_query_nearest_distance()
        logger.info("成功计算最近邻")
        
        # # 测试创建索引
        # table_name = adapter.table_name
        # index_type = "ivfflat"
        # index_params = {
        #     "nlist": 100,
        #     "nprobe": 10
        # }
        # index_info = adapter.create_index(table_name, index_type, index_params)
        # logger.info(f"成功创建索引: {index_info}")
        
        # # 测试性能
        # query_table = adapter._query_table_name
        # topk = 10
        # perf_results = adapter.test_performance(table_name, query_table, topk, index_params)
        # logger.info(f"性能测试结果: QPS={perf_results['qps']}, 平均延迟={perf_results['avg_latency']}ms")
        
        # # 测试计算召回率
        # search_results = perf_results['results']
        # recall_results = adapter.compute_recall(search_results, query_table)
        # logger.info(f"召回率: {recall_results['recall']}")
        
        # # 测试删除索引
        # drop_result = adapter.drop_index(table_name)
        # logger.info(f"删除索引结果: {drop_result}")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
    finally:
        # 关闭连接
        if adapter and adapter.connection:
            adapter.close()
            logger.info("成功关闭数据库连接")
    
    logger.info("OpenGaussAdapter测试完成")

def test_exact_search():
    """测试精确搜索功能"""
    logger.info("开始测试精确搜索功能")
    
    try:
        adapter = OpenGaussAdapter()
        
        # 创建随机向量数据用于测试
        dim = adapter.dimension
        index_count = 1000
        query_count = 10
        index_vectors = np.random.random((index_count, dim)).astype(np.float32)
        query_vectors = np.random.random((query_count, dim)).astype(np.float32)
        
        # 测试精确搜索
        k = 10
        metric = "l2"
        indices, distances = adapter.compute_exact_search(query_vectors, index_vectors, k, metric)
        
        logger.info(f"精确搜索完成，结果形状: indices={indices.shape}, distances={distances.shape}")
        
        # 验证结果
        assert indices.shape == (query_count, k), f"索引结果形状错误: {indices.shape}"
        assert distances.shape == (query_count, k), f"距离结果形状错误: {distances.shape}"
        
        logger.info("精确搜索测试通过")
        
    except Exception as e:
        logger.error(f"精确搜索测试中发生错误: {e}")
    finally:
        if 'adapter' in locals() and adapter and adapter.connection:
            adapter.close()

def test_nearest_neighbors():
    """测试最近邻计算功能"""
    logger.info("开始测试最近邻计算功能")
    
    try:
        adapter = OpenGaussAdapter()
        
        # 测试计算最近邻
        train_table = adapter._sample_table_name
        test_table = adapter._query_table_name
        k = 100
        metric = "l2"
        concurrency = 4
        
        adapter.compute_nearest_neighbors(train_table, test_table, k, metric, concurrency)
        logger.info("成功计算最近邻")
        
        # 获取真实结果
        groundtruth_results = adapter.get_groundtruth_results(test_table)
        logger.info(f"成功获取真实结果，数量: {len(groundtruth_results)}")
        
    except Exception as e:
        logger.error(f"最近邻计算测试中发生错误: {e}")
    finally:
        if 'adapter' in locals() and adapter and adapter.connection:
            adapter.close()

if __name__ == "__main__":
    # 运行测试
    test_opengauss_adapter()
    # test_exact_search()
    # test_nearest_neighbors()
    
    logger.info("所有测试完成")
