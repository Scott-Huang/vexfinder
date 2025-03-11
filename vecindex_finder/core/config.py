#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import multiprocessing
from typing import Dict, Any, Optional, List, Tuple
from core.logging import logger


class Config:
    """配置管理类"""
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
        """
        self.config_path = config_path
        self.parallel_workers = 8
        self.connection_config = {}
        self.table_info_config = {}
        self.sampling_config = {}
        self.query_config = {}
        self.performance_config = {}
        self.index_config = {}
        
        self.load_config()
        # 检查并设置特定配置
        self.check_and_setup_config()
    
    def load_config(self) -> Dict[str, Any]:
        """
        加载OpenGauss配置
        
        Returns:
            配置字典
        """
        if self.config_path is None:
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.config_path = os.path.join(current_dir, 'config.yml')
        
        # 检查配置文件是否存在
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件 {self.config_path} 不存在")
        
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                logger.info(f"成功加载配置文件: {self.config_path}")
                return self.config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    def check_and_setup_config(self) -> None:
        """
        检查并设置特定配置
        
        Raises:
            ValueError: 如果配置文件中缺少必要信息
        """

        # 设置并行工作线程数
        if self.config.get('parallel_workers') is not None:
            self.parallel_workers = int(self.config['parallel_workers'])
        else:
            self.parallel_workers = max(1, int(multiprocessing.cpu_count() * 0.75))
        
        # 检查必要配置项是否存在
        required_sections = ['connection', 'table_info', 'sampling', 'query', 'performance', 'index']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"配置文件中缺少必要部分: {section}")
        
        # 检查connection配置
        required_connection_fields = ['host', 'port', 'user', 'password', 'dbname']
        for field in required_connection_fields:
            if field not in self.config['connection']:
                raise ValueError(f"连接配置中缺少必要字段: {field}")
        
        self.connection_config = self.config['connection']

        # 检查table_info配置
        required_table_info_fields = ['table_name', 'vector_column_name', 'dimension', 'metric']
        for field in required_table_info_fields:
            if field not in self.config['table_info']:
                raise ValueError(f"table_info配置中缺少必要字段: {field}")
        
        self.table_info_config = self.config['table_info']
        self.table_info_config['sample_table_name'] = f"{self.table_info_config['table_name']}_sample_vecindex_finder"
        self.table_info_config['query_table_name'] = f"{self.table_info_config['table_name']}_query_vecindex_finder"
        
        # 检查采样配置
        required_sampling_fields = ['default_ratio', 'min_sample_count', 'max_sample_count']
        for field in required_sampling_fields:
            if field not in self.config['sampling']:
                raise ValueError(f"采样配置中缺少必要字段: {field}")
        
        self.sampling_config = self.config['sampling']

        # 检查query配置
        required_query_fields = ['query_get_type', 'query_data_path']
        for field in required_query_fields:
            if field not in self.config['query']:
                raise ValueError(f"query配置中缺少必要字段: {field}")

        self.query_config = self.config['query']

        # 检查性能配置
        if 'limit' not in self.config['performance'] or self.config['performance']['limit'] is None:
            self.performance_config['limit'] = 100
        else:
            self.performance_config['limit'] = self.config['performance']['limit']
        
        if 'min_recall' not in self.config['performance'] or self.config['performance']['min_recall'] is None:
            self.performance_config['min_recall'] = 0.8
        else:
            self.performance_config['min_recall'] = self.config['performance']['min_recall']

        if 'max_recall' not in self.config['performance'] or self.config['performance']['max_recall'] is None:
            self.performance_config['max_recall'] = 0.999
        else:
            self.performance_config['max_recall'] = self.config['performance']['max_recall']
        
        if 'limit' not in self.config['performance'] or self.config['performance']['limit'] is None:
            self.performance_config['limit'] = 100
        else:
            self.performance_config['limit'] = self.config['performance']['limit']


        # 检查索引配置
        required_index_fields = ['find_index_type', 'auto']
        for field in required_index_fields:
            if field not in self.config['index']:
                raise ValueError(f"索引配置中缺少必要字段: {field}")
        
        self.index_config = self.config['index']

        if not self.index_config['auto']:
            if self.index_config['find_index_type'] not in self.config['index']:
                raise ValueError(f"索引配置中缺少必要字段: {self.index_config['find_index_type']}")



# 创建全局配置实例
config = Config()
