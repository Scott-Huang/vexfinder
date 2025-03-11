#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
import os
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class QueryDatasetManager:
    """查询数据集管理类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化查询数据集管理器
        
        Args:
            config: 配置信息
        """
        self.config = config
        self.default_query_count = config.get("query_count", 10000)
        
    def load_from_json(self, json_path: str) -> np.ndarray:
        """
        从JSON文件加载查询数据集
        
        Args:
            json_path: JSON文件路径
            
        Returns:
            查询向量数据
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON文件不存在: {json_path}")
            
        logger.info(f"从JSON文件加载查询数据: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 检查JSON数据格式
        if isinstance(data, list):
            # 处理列表格式（纯向量列表）
            if not data:
                raise ValueError("JSON数据为空列表")
                
            # 检查数据格式
            if isinstance(data[0], (list, tuple)) or (isinstance(data[0], dict) and "vector" in data[0]):
                # 是向量列表或带有向量字段的字典列表
                vectors = self._extract_vectors_from_data(data)
                logger.info(f"从JSON加载了 {len(vectors)} 个查询向量")
                return np.array(vectors)
            else:
                raise ValueError("JSON数据格式不正确，无法解析为向量")
                
        elif isinstance(data, dict):
            # 处理字典格式
            if "vectors" in data:
                # 包含向量列表的字典
                vectors = data["vectors"]
                logger.info(f"从JSON加载了 {len(vectors)} 个查询向量")
                return np.array(vectors)
            else:
                raise ValueError("JSON字典中未找到'vectors'字段")
        else:
            raise ValueError("无效的JSON查询数据格式")
    
    def _extract_vectors_from_data(self, data: List[Any]) -> List[List[float]]:
        """
        从数据中提取向量
        
        Args:
            data: 数据列表
            
        Returns:
            向量列表
        """
        vectors = []
        
        for item in data:
            if isinstance(item, (list, tuple)):
                # 直接是向量
                vectors.append(item)
            elif isinstance(item, dict) and "vector" in item:
                # 字典中包含向量字段
                vectors.append(item["vector"])
            else:
                logger.warning(f"无法从数据中提取向量: {item}")
                
        return vectors
    
    def sample_from_database(self, db_adapter, table_name: str, 
                           count: Optional[int] = None, 
                           exclude_ids: Optional[List[int]] = None) -> np.ndarray:
        """
        从数据库表中采样查询数据集
        
        Args:
            db_adapter: 数据库适配器
            table_name: 表名
            count: 采样数量，如果为None则使用默认值
            exclude_ids: 需要排除的ID列表
            
        Returns:
            查询向量数据
        """
        count = count or self.default_query_count
        
        logger.info(f"从数据库表 {table_name} 采样 {count} 条数据作为查询集")
        
        # 调用数据库适配器的采样方法
        query_data = db_adapter.sample_query_data(table_name, count, exclude_ids)
        
        logger.info(f"成功采样 {len(query_data)} 条数据作为查询集")
        
        return query_data
        
    def get_query_dataset(self, db_adapter, table_name: str, 
                         user_query_file: Optional[str] = None,
                         count: Optional[int] = None,
                         exclude_ids: Optional[List[int]] = None) -> np.ndarray:
        """
        获取查询数据集（优先使用用户提供的，如果没有则从数据库采样）
        
        Args:
            db_adapter: 数据库适配器
            table_name: 表名
            user_query_file: 用户提供的查询数据JSON文件路径
            count: 采样数量，如果为None则使用默认值
            exclude_ids: 需要排除的ID列表
            
        Returns:
            查询向量数据
        """
        if user_query_file and os.path.exists(user_query_file):
            logger.info(f"使用用户提供的查询数据: {user_query_file}")
            return self.load_from_json(user_query_file)
        else:
            if user_query_file:
                logger.warning(f"用户提供的查询数据文件不存在: {user_query_file}，将从数据库采样")
            else:
                logger.info("未提供查询数据文件，将从数据库采样")
                
            return self.sample_from_database(db_adapter, table_name, count, exclude_ids)
