#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import json
from core.config import config
from core.engine import db_engine
from core.analyzer import Analyzer
from core.logging import logger
from core.sampling import Sampling

config_path = os.path.join(os.path.dirname(__file__), 'config.yml')

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='向量索引参数优化工具')
    
    parser.add_argument('--config', type=str, default='config.yml',
                        help='配置文件路径，默认为config.yml')
    
    parser.add_argument('--output-dir', type=str, default='./cursor/index_finder_results',
                        help='输出目录路径，默认为./cursor/index_finder_results')
    
    parser.add_argument('--skip-sampling', action='store_true',
                        help='跳过数据采样步骤，直接使用已有的采样数据')
    
    parser.add_argument('--skip-distance-computation', action='store_true',
                        help='跳过距离计算步骤，直接使用已有的距离数据')
    
    parser.add_argument('--parallel-workers', type=int,
                        help='并行工作线程数，不指定则使用配置文件中的设置')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    try:
        # 加载配置
        logger.info(f"加载配置文件: {args.config}")
        config.prepare_with_yaml(args.config)
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 如果命令行指定了并行工作线程数，覆盖配置文件中的设置
        if args.parallel_workers:
            config.parallel_workers = args.parallel_workers
            logger.info(f"使用命令行指定的并行工作线程数: {args.parallel_workers}")
        
        # 创建采样器
        logger.info("创建采样器")
        sampling = Sampling()
        
        # 数据采样
        if not args.skip_sampling:
            logger.info("开始基础数据采样")
            sample_table = sampling.sampling_data()
            logger.info(f"基础数据采样完成，采样表: {sample_table}")
            logger.info("开始查询数据采样")
            query_table = sampling.sampling_query_data()
            logger.info(f"查询数据采样完成，查询表: {query_table}")
        else:
            logger.info("跳过数据采样步骤，使用已有的采样数据")
            config.table_info.sample_table_count = sampling.get_sample_table_count()
            config.table_info.query_table_count = sampling.get_query_table_count()
        
        
        # 计算距离
        if not args.skip_distance_computation:
            logger.info("开始计算最近邻距离")
            sampling.compute_sample_query_distance()
            logger.info("最近邻距离计算完成")
        else:
            logger.info("跳过距离计算步骤")
        
        query_data = sampling.get_all_query_data()
        logger.info(f"查询数据共有: {len(query_data)} 条")
        
        
        # 创建索引分析器
        logger.info("创建索引分析器")
        analyzer = Analyzer(config.index_config, db_engine, query_data, config.performance, config.table_info)
        
        # 查找最佳索引参数
        logger.info("开始查找最佳索引参数")
        best_result = analyzer.analyze()
        logger.info(f"找到最佳索引参数: {json.dumps(best_result, indent=4)}")

        
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 