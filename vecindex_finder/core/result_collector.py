#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import datetime
from typing import Dict, List, Any, Optional, Tuple
from core.config import Config
from core.logging import logger

class ResultCollector:
    """结果收集器，负责收集和分析测试结果"""
    
    def __init__(self, index_type: str):
        """
        初始化结果收集器
        
        Args:
            index_type: 索引类型
        """
        self.index_type = index_type
        # 存储所有测试结果
        self.results = []
        
        # 创建结果存储目录
        if not os.path.isdir(os.path.join("results")):
            os.makedirs(os.path.join("results"))

        # 创建索引类型_日期时间格式的文件夹
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.result_dir = os.path.join("results", f"{index_type}_{timestamp}")
        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir)
    
        
    def add_best_result(self, result: Dict[str, Any]) -> None:
        """
        添加测试结果
        
        Args:
            result: 测试结果
        """
        self.results.append(result)
        self.store_best_result(result)
        logger.info(f"添加测试结果: 索引类型={result['index_type']}, QPS={result['best_performance']['qps']:.2f}, 召回率={result['best_performance']['recall']:.4f}")
    
    
    def store_best_result(self, result: Dict[str, Any]) -> str:
        """
        存储最佳实践结果
        Args:
            result: 测试结果
        Returns:
            存储路径
        """
        index_param = result['index_param']
        best_param = result['best_query_param']
        # 构造文件名：索引参数+搜索参数
        index_param_str = "_".join([f"{k}_{v}" for k, v in sorted(index_param.items())])
        query_param_str = "_".join([f"{k}_{v}" for k, v in sorted(best_param.items())])
        filename = f"{index_param_str}_{query_param_str}.json"
        filepath = os.path.join(self.result_dir, filename)
        # 写入JSON文件
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"最佳实践结果已保存到: {filepath}")
        return filepath
        
    
    def find_best_params(self) -> Dict[str, Any]:
        """
        找出最佳索引参数
        
        Returns:
            最佳索引参数和性能结果
        """
        if not self.results:
            logger.warning("没有测试结果可分析")
            return {}
        
        # 筛选满足最低召回率要求的结果
        valid_results = [r for r in self.results if r['avg_recall'] >= self.min_recall]
        
        if not valid_results:
            logger.warning(f"没有找到满足最低召回率 {self.min_recall} 的索引参数")
            # 如果没有满足最低召回率的结果，返回召回率最高的结果
            return max(self.results, key=lambda x: x['avg_recall'])
        
        # 在满足召回率要求的结果中，找出QPS最高的
        best_result = max(valid_results, key=lambda x: x['qps'])
        
        logger.info(f"找到最佳索引参数: {best_result}")
        return best_result
    
    def find_best_params_by_type(self) -> Dict[str, Dict[str, Any]]:
        """
        按索引类型找出最佳参数
        
        Returns:
            按索引类型分组的最佳参数
        """
        if not self.results:
            logger.warning("没有测试结果可分析")
            return {}
        
        # 按索引类型分组
        results_by_type = {}
        for result in self.results:
            index_type = result['index_type']
            if index_type not in results_by_type:
                results_by_type[index_type] = []
            results_by_type[index_type].append(result)
        
        # 对每种索引类型找出最佳参数
        best_params_by_type = {}
        for index_type, type_results in results_by_type.items():
            # 筛选满足最低召回率要求的结果
            valid_results = [r for r in type_results if r['avg_recall'] >= self.min_recall]
            
            if not valid_results:
                logger.warning(f"索引类型 {index_type} 没有找到满足最低召回率 {self.min_recall} 的参数")
                # 如果没有满足最低召回率的结果，返回召回率最高的结果
                best_params_by_type[index_type] = max(type_results, key=lambda x: x['avg_recall'])
            else:
                # 在满足召回率要求的结果中，找出QPS最高的
                best_params_by_type[index_type] = max(valid_results, key=lambda x: x['qps'])
        
        return best_params_by_type
    
    def export_results(self, output_path: str) -> None:
        """
        将测试结果导出到CSV文件
        
        Args:
            output_path: 输出文件路径
        """
        if not self.results:
            logger.warning("没有测试结果可导出")
            return
        
        # 将结果转换为DataFrame
        df = pd.DataFrame(self.results)
        
        # 保存到CSV
        df.to_csv(output_path, index=False)
        logger.info(f"测试结果已导出到 {output_path}")
        
        # 同时保存为JSON格式
        json_path = output_path.replace('.csv', '.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"测试结果已导出到 {json_path}")
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        生成测试结果摘要
        
        Returns:
            测试结果摘要
        """
        if not self.results:
            logger.warning("没有测试结果可生成摘要")
            return {}
        
        # 获取最佳参数
        best_params = self.find_best_params()
        
        # 按索引类型获取最佳参数
        best_params_by_type = self.find_best_params_by_type()
        
        # 计算平均QPS和召回率
        avg_qps = sum(r['qps'] for r in self.results) / len(self.results)
        avg_recall = sum(r['avg_recall'] for r in self.results) / len(self.results)
        
        # 统计各索引类型的数量
        index_type_counts = {}
        for result in self.results:
            index_type = result['index_type']
            if index_type not in index_type_counts:
                index_type_counts[index_type] = 0
            index_type_counts[index_type] += 1
        
        # 生成摘要
        summary = {
            'total_tests': len(self.results),
            'avg_qps': avg_qps,
            'avg_recall': avg_recall,
            'index_type_counts': index_type_counts,
            'best_params': best_params,
            'best_params_by_type': best_params_by_type
        }
        
        return summary
    
    def export_summary(self, output_path: str) -> None:
        """
        将测试结果摘要导出到JSON文件
        
        Args:
            output_path: 输出文件路径
        """
        summary = self.generate_summary()
        
        if not summary:
            logger.warning("没有测试结果摘要可导出")
            return
        
        # 保存为JSON格式
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"测试结果摘要已导出到 {output_path}")
        
        # 同时生成一个简单的文本报告
        txt_path = output_path.replace('.json', '.txt')
        with open(txt_path, 'w') as f:
            f.write("向量索引参数优化测试结果摘要\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"总测试数: {summary['total_tests']}\n")
            f.write(f"平均QPS: {summary['avg_qps']:.2f}\n")
            f.write(f"平均召回率: {summary['avg_recall']:.4f}\n\n")
            
            f.write("索引类型统计:\n")
            for index_type, count in summary['index_type_counts'].items():
                f.write(f"  {index_type}: {count}\n")
            f.write("\n")
            
            f.write("最佳索引参数:\n")
            best = summary['best_params']
            f.write(f"  索引类型: {best['index_type']}\n")
            f.write(f"  QPS: {best['qps']:.2f}\n")
            f.write(f"  召回率: {best['avg_recall']:.4f}\n")
            if 'index_creation_time' in best:
                f.write(f"  索引创建时间: {best['index_creation_time']:.2f} 秒\n")
            f.write("\n")
            
            f.write("各索引类型最佳参数:\n")
            for index_type, params in summary['best_params_by_type'].items():
                f.write(f"  {index_type}:\n")
                f.write(f"    QPS: {params['qps']:.2f}\n")
                f.write(f"    召回率: {params['avg_recall']:.4f}\n")
                if 'index_creation_time' in params:
                    f.write(f"    索引创建时间: {params['index_creation_time']:.2f} 秒\n")
                f.write("\n")
        
        logger.info(f"测试结果摘要文本报告已导出到 {txt_path}") 