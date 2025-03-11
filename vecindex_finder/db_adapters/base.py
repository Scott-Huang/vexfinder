#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Any, Tuple, Optional


class VectorDatabaseAdapter(ABC):
    """向量数据库适配器基类，提供统一的接口"""
    
    @abstractmethod
    def connect(self, config: Dict[str, Any]):
        """
        连接到数据库
        
        Args:
            config: 连接配置，包含主机、端口等信息
            
        Returns:
            self: 返回自身实例以支持链式调用
        """
        pass
        
    @abstractmethod
    def sample_data(self, table_name: str, sample_ratio: float, method: str = "random") -> np.ndarray:
        """
        从指定表采样数据（用于创建索引）
        
        Args:
            table_name: 表名
            sample_ratio: 采样比例 (0-1)
            method: 采样方法，支持 "random", "stratified", "cluster"
            
        Returns:
            采样的向量数据
        """
        pass
        
    @abstractmethod
    def sample_query_data(self, table_name: str, count: int, exclude_ids: Optional[List[int]] = None) -> np.ndarray:
        """
        采样查询数据（用于测试）
        
        Args:
            table_name: 表名
            count: 采样数量
            exclude_ids: 需要排除的ID列表（通常是已用于索引的数据ID）
            
        Returns:
            查询向量数据
        """
        pass
        
    @abstractmethod
    def compute_exact_search(self, query_vectors: np.ndarray, index_vectors: np.ndarray, 
                             k: int, metric: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算精确搜索结果（暴力搜索）
        
        Args:
            query_vectors: 查询向量
            index_vectors: 索引向量（用于搜索的向量集合）
            k: 返回的最近邻数量
            metric: 距离度量类型 ('l2', 'ip', 'cosine')
            
        Returns:
            (indices, distances): 索引和距离的元组
                indices: 形状为 (len(query_vectors), k) 的整数数组
                distances: 形状为 (len(query_vectors), k) 的浮点数数组
        """
        pass
        
    @abstractmethod
    def create_index(self, table_name: str, sampled_data: np.ndarray, 
                    index_type: str, index_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建指定参数的索引
        
        Args:
            table_name: 表名
            sampled_data: 用于创建索引的采样数据
            index_type: 索引类型，如 'ivfflat', 'hnsw'
            index_params: 索引参数
            
        Returns:
            包含索引信息的字典，如临时表名、索引创建时间等
        """
        pass
        
    @abstractmethod
    def test_performance(self, table_name: str, query_data: np.ndarray, topk: int, 
                        index_params: Dict[str, Any], concurrency: int = 1) -> Dict[str, Any]:
        """
        测试索引性能（QPS和延迟）
        
        Args:
            table_name: 表名
            query_data: 查询数据
            topk: 查询返回的邻居数量
            index_params: 索引参数
            concurrency: 并发查询数
            
        Returns:
            包含性能指标的字典，如QPS、平均延迟等
        """
        pass
        
    @abstractmethod
    def compute_recall(self, search_results: List[List[int]], 
                     groundtruth_results: List[List[int]]) -> float:
        """
        计算召回率
        
        Args:
            search_results: 索引搜索结果，每个查询的topk结果ID
            groundtruth_results: 精确搜索结果，每个查询的topk结果ID
            
        Returns:
            召回率 (0-1)
        """
        pass
        
    @abstractmethod
    def drop_index(self, table_name: str) -> bool:
        """
        删除索引
        
        Args:
            table_name: 表名
            
        Returns:
            操作是否成功
        """
        pass
        
    @abstractmethod
    def close(self) -> None:
        """
        关闭数据库连接
        """
        pass
