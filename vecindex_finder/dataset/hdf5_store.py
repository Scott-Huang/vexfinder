#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import numpy as np
import os
from typing import Tuple, List, Dict, Any, Optional
import time
import logging

logger = logging.getLogger(__name__)

class HDF5DatasetStore:
    """HDF5数据集存储管理类"""
    
    def __init__(self, output_dir: str):
        """
        初始化HDF5数据存储
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def create_dataset_file(self, name: str, train_data: np.ndarray, 
                           test_data: np.ndarray, distance: str, 
                           dimension: int) -> str:
        """
        创建包含训练和测试数据的HDF5文件
        
        Args:
            name: 数据集名称
            train_data: 训练数据（用于创建索引的采样数据）
            test_data: 测试数据（查询数据）
            distance: 距离度量类型
            dimension: 向量维度
            
        Returns:
            HDF5文件路径
        """
        output_path = os.path.join(self.output_dir, f"{name}.hdf5")
        
        logger.info(f"创建数据集文件: {output_path}")
        logger.info(f"训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")
        
        with h5py.File(output_path, 'w') as f:
            # 存储训练数据（用于构建索引的采样数据）
            f.create_dataset('train', data=train_data)
            
            # 存储测试数据（查询数据）
            f.create_dataset('test', data=test_data)
            
            # 存储元数据
            f.attrs['distance'] = distance
            f.attrs['dimension'] = dimension
            f.attrs['count'] = len(train_data)
            f.attrs['test_count'] = len(test_data)
            f.attrs['created_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
            
        logger.info(f"数据集文件创建成功: {output_path}")
        return output_path
        
    def compute_groundtruth(self, dataset_path: str, neighbors: int = 100, 
                          batch_size: int = 1000) -> str:
        """
        计算并存储暴力搜索的精确结果
        
        Args:
            dataset_path: 数据集文件路径
            neighbors: 每个查询返回的邻居数量
            batch_size: 计算批次大小，用于优化内存使用
            
        Returns:
            groundtruth文件路径
        """
        gt_path = os.path.join(self.output_dir, 
                               os.path.basename(dataset_path).replace('.hdf5', '.gt.hdf5'))
        
        logger.info(f"计算groundtruth: {gt_path}")
        
        # 加载数据集
        with h5py.File(dataset_path, 'r') as f:
            train = f['train'][:]
            test = f['test'][:]
            distance = f.attrs['distance']
            
        total_queries = len(test)
        
        # 创建结果文件
        with h5py.File(gt_path, 'w') as f:
            neighbors_dset = f.create_dataset('neighbors', shape=(total_queries, neighbors), dtype='i')
            distances_dset = f.create_dataset('distances', shape=(total_queries, neighbors), dtype='f')
            
            # 批量计算，避免内存溢出
            for i in range(0, total_queries, batch_size):
                end_idx = min(i + batch_size, total_queries)
                current_batch = test[i:end_idx]
                
                logger.info(f"计算查询批次 {i+1}-{end_idx} / {total_queries}")
                
                # 根据距离类型计算
                if distance == 'l2' or distance == 'euclidean':
                    # 欧几里得距离计算
                    batch_neighbors, batch_distances = self._compute_euclidean(current_batch, train, neighbors)
                elif distance == 'ip' or distance == 'inner_product':
                    # 内积距离计算
                    batch_neighbors, batch_distances = self._compute_inner_product(current_batch, train, neighbors)
                elif distance == 'cosine' or distance == 'angular':
                    # 余弦距离计算
                    batch_neighbors, batch_distances = self._compute_cosine(current_batch, train, neighbors)
                else:
                    raise ValueError(f"不支持的距离类型: {distance}")
                
                # 保存当前批次结果
                neighbors_dset[i:end_idx] = batch_neighbors
                distances_dset[i:end_idx] = batch_distances
            
            # 保存元数据
            f.attrs['distance'] = distance
            f.attrs['neighbors'] = neighbors
            f.attrs['created_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        logger.info(f"Groundtruth计算完成: {gt_path}")
        return gt_path
    
    def _compute_euclidean(self, queries: np.ndarray, 
                          corpus: np.ndarray, 
                          k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算欧几里得距离的最近邻
        
        Args:
            queries: 查询向量
            corpus: 语料库向量
            k: 返回的邻居数量
            
        Returns:
            (indices, distances): 索引和距离数组
        """
        neighbors = np.zeros((len(queries), k), dtype=np.int32)
        distances = np.zeros((len(queries), k), dtype=np.float32)
        
        for i, query in enumerate(queries):
            # 计算到所有点的距离
            dist = np.sum((corpus - query) ** 2, axis=1)
            
            # 获取最近的k个点
            idx = np.argsort(dist)[:k]
            neighbors[i] = idx
            distances[i] = dist[idx]
        
        return neighbors, distances
    
    def _compute_inner_product(self, queries: np.ndarray, 
                              corpus: np.ndarray, 
                              k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算内积（点积）距离的最近邻
        注意：内积越大表示越相似，所以这里取负值排序
        
        Args:
            queries: 查询向量
            corpus: 语料库向量
            k: 返回的邻居数量
            
        Returns:
            (indices, distances): 索引和距离数组
        """
        neighbors = np.zeros((len(queries), k), dtype=np.int32)
        distances = np.zeros((len(queries), k), dtype=np.float32)
        
        for i, query in enumerate(queries):
            # 计算内积
            dist = -np.dot(corpus, query)  # 负值，使得排序结果是最大内积在前
            
            # 获取内积最大的k个点
            idx = np.argsort(dist)[:k]
            neighbors[i] = idx
            distances[i] = dist[idx]
        
        return neighbors, distances
    
    def _compute_cosine(self, queries: np.ndarray, 
                       corpus: np.ndarray, 
                       k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算余弦距离的最近邻
        
        Args:
            queries: 查询向量
            corpus: 语料库向量
            k: 返回的邻居数量
            
        Returns:
            (indices, distances): 索引和距离数组
        """
        neighbors = np.zeros((len(queries), k), dtype=np.int32)
        distances = np.zeros((len(queries), k), dtype=np.float32)
        
        # 预先计算语料库向量的模
        corpus_norms = np.linalg.norm(corpus, axis=1)
        
        for i, query in enumerate(queries):
            # 计算查询向量的模
            query_norm = np.linalg.norm(query)
            
            # 计算余弦相似度并转换为距离（1-相似度）
            similarity = np.dot(corpus, query) / (corpus_norms * query_norm)
            dist = 1 - similarity
            
            # 获取距离最小的k个点
            idx = np.argsort(dist)[:k]
            neighbors[i] = idx
            distances[i] = dist[idx]
        
        return neighbors, distances
    
    def load_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        加载数据集
        
        Args:
            dataset_path: 数据集文件路径
            
        Returns:
            包含数据集信息的字典
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
        
        with h5py.File(dataset_path, 'r') as f:
            train = f['train'][:]
            test = f['test'][:]
            distance = f.attrs['distance']
            dimension = f.attrs['dimension']
            count = f.attrs['count']
            test_count = f.attrs['test_count']
        
        return {
            'train': train,
            'test': test,
            'distance': distance,
            'dimension': dimension,
            'count': count,
            'test_count': test_count
        }
    
    def load_groundtruth(self, gt_path: str) -> Dict[str, Any]:
        """
        加载groundtruth数据
        
        Args:
            gt_path: groundtruth文件路径
            
        Returns:
            包含groundtruth信息的字典
        """
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Groundtruth文件不存在: {gt_path}")
        
        with h5py.File(gt_path, 'r') as f:
            neighbors = f['neighbors'][:]
            distances = f['distances'][:]
            distance = f.attrs['distance']
            k = f.attrs['neighbors']
        
        return {
            'neighbors': neighbors,
            'distances': distances,
            'distance': distance,
            'k': k
        }
