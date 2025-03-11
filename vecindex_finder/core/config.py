#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class Config:
    """配置管理类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
        """
        self.config_path = config_path
        self.config = {}
        self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """
        加载配置
        
        Returns:
            配置字典
        """
        # 加载默认配置
        default_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                         'config', 'default.yml')
                                         
        if os.path.exists(default_config_path):
            with open(default_config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
                logger.info(f"已加载默认配置: {default_config_path}")
        else:
            logger.warning(f"默认配置文件不存在: {default_config_path}")
            self.config = {}
            
        # 如果指定了配置文件，则加载并覆盖默认配置
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f) or {}
                
            # 合并配置
            self.merge_config(self.config, user_config)
            logger.info(f"已加载用户配置: {self.config_path}")
            
        return self.config
        
    def merge_config(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并配置
        
        Args:
            base: 基础配置
            override: 覆盖配置
            
        Returns:
            合并后的配置
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                # 如果两边都是字典，则递归合并
                self.merge_config(base[key], value)
            else:
                # 否则直接覆盖
                base[key] = value
                
        return base
        
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项
        
        Args:
            key: 配置键，支持点号分隔的嵌套路径，如 'sampling.ratio'
            default: 默认值
            
        Returns:
            配置值
        """
        if '.' not in key:
            return self.config.get(key, default)
            
        # 处理嵌套配置
        parts = key.split('.')
        current = self.config
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
                
        return current
        
    def load_index_params(self, index_type: str) -> Dict[str, Any]:
        """
        加载特定索引类型的参数
        
        Args:
            index_type: 索引类型，如 'ivfflat', 'hnsw'
            
        Returns:
            索引参数配置
        """
        params_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                 'config', 'index_params', f"{index_type}.yml")
                                 
        if not os.path.exists(params_file):
            logger.warning(f"索引参数文件不存在: {params_file}，将使用默认参数")
            return {}
            
        with open(params_file, 'r', encoding='utf-8') as f:
            params = yaml.safe_load(f) or {}
            
        logger.info(f"已加载索引参数: {index_type}")
        return params
        
    def __str__(self) -> str:
        """字符串表示"""
        return yaml.dump(self.config, default_flow_style=False, allow_unicode=True)
        
    def __repr__(self) -> str:
        """对象表示"""
        return f"Config(path={self.config_path}, items={len(self.config)})"
