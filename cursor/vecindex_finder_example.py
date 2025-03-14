#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
向量索引参数优化工具使用示例
"""

import os
import sys
import time
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vecindex_finder.core.config import Config
from vecindex_finder.core.engine import DatabaseEngine
from vecindex_finder.core.analyzer import IndexAnalyzer
from vecindex_finder.core.visualizer import IndexVisualizer
from vecindex_finder.core.logging import logger
from vecindex_finder.sampling import Sampling


def main():
    """示例主函数"""
    # 设置配置文件路径
    config_path = "../vecindex_finder/config.yml"
    
    # 设置输出目录
    output_dir = "./index_finder_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 加载配置
        logger.info(f"加载配置文件: {config_path}")
        config = Config(config_path)
        
        # 创建数据库引擎
        logger.info("创建数据库引擎")
        db_engine = DatabaseEngine(config)
        
        # 创建采样器
        logger.info("创建采样器")
        sampling = Sampling(config, db_engine)
        
        # 数据采样
        logger.info("开始数据采样")
        sample_table = sampling.sampling_data()
        logger.info(f"数据采样完成，采样表: {sample_table}")
        
        # 查询采样
        logger.info("开始查询采样")
        query_table = sampling.sampling_query_data()
        logger.info(f"查询采样完成，查询表: {query_table}")
        
        # 计算距离
        logger.info("开始计算最近邻距离")
        sampling.compute_sample_query_distance()
        logger.info("最近邻距离计算完成")
        
        # 创建索引分析器
        logger.info("创建索引分析器")
        analyzer = IndexAnalyzer(config, db_engine, sampling)
        
        # 查找最佳索引参数
        logger.info("开始查找最佳索引参数")
        best_params = analyzer.find_best_index_params()
        logger.info(f"找到最佳索引参数: {best_params}")
        
        # 导出结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f"index_results_{timestamp}.csv")
        analyzer.export_results(results_file)
        
        # 创建可视化工具
        logger.info("创建可视化工具")
        visualizer = IndexVisualizer(results_data=analyzer.results)
        
        # 生成报告
        report_dir = os.path.join(output_dir, f"report_{timestamp}")
        logger.info(f"生成性能报告到目录: {report_dir}")
        visualizer.generate_report(report_dir)
        
        # 记录总耗时
        total_time = time.time() - start_time
        logger.info(f"索引参数优化完成，总耗时: {total_time:.2f} 秒")
        
        # 打印最佳参数
        print("\n" + "="*50)
        print("最佳索引参数:")
        print(f"索引类型: {best_params['index_type']}")
        
        if best_params['index_type'] == 'ivfflat':
            print(f"nlist: {best_params['nlist']}")
            print(f"nprobe: {best_params['nprobe']}")
        elif best_params['index_type'] == 'ivfpq':
            print(f"nlist: {best_params['nlist']}")
            print(f"m: {best_params['m']}")
            print(f"nbits: {best_params['nbits']}")
            print(f"nprobe: {best_params['nprobe']}")
            if 'refine_factor' in best_params:
                print(f"refine_factor: {best_params['refine_factor']}")
        elif best_params['index_type'] == 'hnsw':
            print(f"M: {best_params['M']}")
            print(f"efConstruction: {best_params['efConstruction']}")
            print(f"ef: {best_params['ef']}")
        
        print(f"QPS: {best_params['qps']:.2f}")
        print(f"平均召回率: {best_params['avg_recall']:.4f}")
        if 'index_creation_time' in best_params:
            print(f"索引创建时间: {best_params['index_creation_time']:.2f} 秒")
        print("="*50)
        
        print(f"\n详细报告已生成到目录: {report_dir}")
        
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 