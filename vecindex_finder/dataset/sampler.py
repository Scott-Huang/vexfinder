#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
import time
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class DataSampler:
    """数据采样器类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据采样器
        
        Args:
            config: 配置信息
        """
        self.config = config
        self.sampling_config = config.get("sampling", {})
        self.default_ratio = self.sampling_config.get("index_ratio", 0.1)
        self.default_method = self.sampling_config.get("method", "random")
        self.seed = self.sampling_config.get("seed", 42)
        
    def sample(self, db_adapter, table_name: str, ratio: Optional[float] = None, 
              method: Optional[str] = None) -> Dict[str, Any]:
        """
        从数据库表中采样数据
        
        Args:
            db_adapter: 数据库适配器
            table_name: 表名
            ratio: 采样比例，如果为None则使用默认值
            method: 采样方法，如果为None则使用默认值
            
        Returns:
            包含采样数据和元信息的字典
        """
        ratio = ratio or self.default_ratio
        method = method or self.default_method
        
        if ratio <= 0 or ratio > 1:
            raise ValueError(f"采样比例必须在(0,1]范围内，当前值: {ratio}")
        
        np.random.seed(self.seed)
        
        logger.info(f"从表 {table_name} 采样 {ratio*100:.1f}% 的数据，方法: {method}")
        
        # 调用数据库适配器的采样方法
        start_time = time.time()
        sampled_data = db_adapter.sample_data(table_name, ratio, method)
        sample_time = time.time() - start_time
        
        # 获取原始表大小和采样数据大小
        original_size = db_adapter.get_table_size(table_name)
        sampled_size = len(sampled_data)
        
        logger.info(f"采样完成，耗时: {sample_time:.2f}秒")
        logger.info(f"原始数据大小: {original_size}, 采样数据大小: {sampled_size}")
        
        # 返回采样结果
        return {
            'data': sampled_data,
            'original_size': original_size,
            'sampled_size': sampled_size,
            'ratio': ratio,
            'method': method,
            'sample_time': sample_time
        }
        
    def save_sample_info(self, output_dir: str, sample_info: Dict[str, Any], table_name: str) -> str:
        """
        保存采样信息到文件
        
        Args:
            output_dir: 输出目录
            sample_info: 采样信息
            table_name: 表名
            
        Returns:
            信息文件路径
        """
        import os
        import json
        import time
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建不包含数据的信息字典
        info_dict = {
            'table_name': table_name,
            'original_size': sample_info['original_size'],
            'sampled_size': sample_info['sampled_size'],
            'ratio': sample_info['ratio'],
            'method': sample_info['method'],
            'sample_time': sample_info['sample_time'],
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 保存到JSON文件
        file_path = os.path.join(output_dir, f"{table_name}_sample_info.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(info_dict, f, indent=2, ensure_ascii=False)
            
        logger.info(f"采样信息已保存到: {file_path}")
        
        return file_path
