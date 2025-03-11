#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
import importlib
from typing import Dict, Any, List, Optional

from vecindex_finder.core.config import Config
from vecindex_finder.dataset.sampler import DataSampler
from vecindex_finder.dataset.query_set import QueryDatasetManager
from vecindex_finder.dataset.hdf5_store import HDF5DatasetStore

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('cursor/index_finder.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='向量数据库索引参数推荐工具')
    
    # 数据库参数
    parser.add_argument('--db', type=str, required=True, help='向量数据库类型 (milvus, opengauss, pgvector)')
    parser.add_argument('--table', type=str, required=True, help='数据表名称')
    parser.add_argument('--dim', type=int, help='向量维度')
    parser.add_argument('--metric', type=str, choices=['l2', 'ip', 'cosine'], default='l2', 
                      help='距离度量类型 (l2, ip, cosine)')
    
    # 索引参数
    parser.add_argument('--indices', type=str, help='要测试的索引类型列表，逗号分隔 (ivfflat,hnsw,ivfpq,diskann)')
    
    # 采样参数
    parser.add_argument('--sample-ratio', type=float, help='索引数据采样比例 (0-1)')
    parser.add_argument('--sample-method', type=str, choices=['random', 'stratified', 'cluster'], 
                      help='采样方法')
    
    # 查询数据集参数
    parser.add_argument('--query-json', type=str, help='查询数据JSON文件路径')
    parser.add_argument('--query-count', type=int, help='自动采样的查询数量')
    
    # 测试参数
    parser.add_argument('--topk', type=int, help='查询返回的结果数')
    parser.add_argument('--concurrency', type=int, help='并发查询数')
    
    # 输出参数
    parser.add_argument('--output', type=str, help='输出目录')
    
    # 配置文件
    parser.add_argument('--config', type=str, help='配置文件路径')
    
    # 优化目标
    parser.add_argument('--optimize-for', type=str, choices=['qps', 'recall', 'combined'], 
                      default='combined', help='优化目标')
    parser.add_argument('--weight-qps', type=float, help='QPS权重 (0-1)')
    parser.add_argument('--weight-recall', type=float, help='召回率权重 (0-1)')
    
    # 其他参数
    parser.add_argument('--verbose', action='store_true', help='详细输出模式')
    
    return parser.parse_args()

def load_db_adapter(db_type: str, config: Dict[str, Any]):
    """
    加载数据库适配器
    
    Args:
        db_type: 数据库类型
        config: 配置
        
    Returns:
        数据库适配器实例
    """
    try:
        # 动态导入适配器模块
        module_path = f"vecindex_finder.db_adapters.{db_type.lower()}"
        adapter_module = importlib.import_module(f"{module_path}.adapter")
        
        # 获取适配器类，约定为 XxxAdapter 格式
        adapter_class_name = "".join(word.capitalize() for word in db_type.split('_')) + "Adapter"
        adapter_class = getattr(adapter_module, adapter_class_name)
        
        # 创建适配器实例并连接
        db_config = config.get(f"database.{db_type.lower()}", {})
        adapter = adapter_class().connect(db_config)
        
        logger.info(f"已加载数据库适配器: {db_type}")
        return adapter
        
    except (ImportError, AttributeError) as e:
        logger.error(f"加载数据库适配器失败: {e}")
        raise ValueError(f"不支持的数据库类型: {db_type}")

def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 加载配置
        config_obj = Config(args.config)
        config = config_obj.config
        
        # 更新配置（命令行参数优先）
        if args.output:
            config["test"]["output_dir"] = args.output
        if args.sample_ratio:
            config["sampling"]["index_ratio"] = args.sample_ratio
        if args.sample_method:
            config["sampling"]["method"] = args.sample_method
        if args.query_json:
            config["query_dataset"]["source"] = "user_json"
            config["query_dataset"]["json_path"] = args.query_json
        if args.query_count:
            config["query_dataset"]["query_count"] = args.query_count
        if args.topk:
            config["performance"]["topk"] = args.topk
        if args.concurrency:
            config["performance"]["concurrency"] = [args.concurrency]
        if args.weight_qps:
            config["performance"]["weights"]["qps"] = args.weight_qps
        if args.weight_recall:
            config["performance"]["weights"]["recall"] = args.weight_recall
            
        # 创建输出目录
        output_dir = config["test"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据库适配器
        db_adapter = load_db_adapter(args.db, config)
        
        # 设置要测试的索引
        indices = args.indices.split(',') if args.indices else []
        if not indices:
            logger.warning("未指定索引类型，将使用所有支持的索引")
            # 这里可以添加默认索引列表
            
        # 创建采样器
        sampler = DataSampler(config)
        
        # 从表中采样数据
        sample_info = sampler.sample(db_adapter, args.table)
        sampled_data = sample_info["data"]
        
        # 创建查询数据集管理器
        query_manager = QueryDatasetManager(config["query_dataset"])
        
        # 获取查询数据集
        query_data = query_manager.get_query_dataset(
            db_adapter, 
            args.table, 
            config["query_dataset"].get("json_path"),
            config["query_dataset"].get("query_count")
        )
        
        # 创建HDF5数据存储
        hdf5_store = HDF5DatasetStore(output_dir)
        
        # 创建数据集文件
        dataset_path = hdf5_store.create_dataset_file(
            f"{args.table}_sample",
            sampled_data,
            query_data,
            args.metric,
            args.dim or sampled_data.shape[1]
        )
        
        # 计算groundtruth（暴力搜索结果）
        gt_path = hdf5_store.compute_groundtruth(
            dataset_path,
            config["performance"]["topk"]
        )
        
        # TODO: 实现索引测试引擎，测试不同索引参数
        
        # TODO: 实现结果分析模块，推荐最优参数
        
        logger.info("索引参数推荐完成")
        
    except Exception as e:
        logger.error(f"执行过程中发生错误: {e}", exc_info=True)
        return 1
        
    return 0

if __name__ == '__main__':
    sys.exit(main())
