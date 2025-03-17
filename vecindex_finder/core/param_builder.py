#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Any, Tuple, Literal
from core.logging import logger
import math
from core.types import IndexAndQueryParam

class IndexParamBuilder:
    """索引参数构建器，负责生成不同索引类型的参数组合"""
    
    def __init__(self, sample_table_count: int, dimension: int, index_type: str):
        """
        初始化索引参数构建器
        
        Args:
            sample_table_count: 样本数量
            dimension: 数据维度
            index_type: 索引类型，如'ivfflat', 'ivfpq', 'hnsw', 'diskann'等
        """
        self.sample_table_count = sample_table_count
        self.dimension = dimension
        self.index_type = index_type

    
    def get_theoretical_param(self) -> IndexAndQueryParam:
        """
        获取理论最优的索引参数和查询参数
        
        根据数据维度和行数计算理论最优参数。
        根据index_type返回对应的参数。
        
        Returns:
            IndexParam对象
        """
        # 使用类成员变量
        sample_count = self.sample_table_count
        index_type = self.index_type
        
        # 如果未指定索引类型，默认使用ivfflat
        if not index_type:
            index_type = 'ivfflat'
        
        # IVF-FLAT理论参数
        if index_type == 'ivfflat':
            # 根据数据量调整nlist值
            base_nlist = int(math.sqrt(sample_count))
            
            # 小数据集，使用较小的nlist以获得更好的精度
            if sample_count < 10000:
                base_nlist = max(16, min(base_nlist, 64))
            # 大数据集，使用略大的nlist以提高性能
            elif sample_count > 1000000:
                base_nlist = max(1000, int(base_nlist * 1.2))
            
            # 根据数据量调整nprobe比例
            nprobe_ratio = 0.05  # 默认为5%
            if sample_count < 10000:
                nprobe_ratio = 0.1  # 小数据集使用更大的比例
            elif sample_count > 1000000:
                nprobe_ratio = 0.03  # 大数据集使用更小的比例
            
            nprobe_value = max(1, int(base_nlist * nprobe_ratio))
            
            return IndexAndQueryParam(
                index_type='ivfflat',
                index_param={
                    'ivf_nlist': base_nlist,
                    'ivf_nlist_range': [1, 10000]  
                },
                query_param={
                    'ivf_probes': nprobe_value,
                    'ivf_probes_range': [1, 10000]
                }
            )
        
        # IVF-PQ理论参数
        elif index_type == 'ivfpq':
            # 根据数据量调整nlist值
            base_nlist = int(math.sqrt(sample_count))
            
            # 小数据集，使用较小的nlist以获得更好的精度
            if sample_count < 10000:
                base_nlist = max(16, min(base_nlist, 64))
            # 大数据集，使用略大的nlist以提高性能
            elif sample_count > 1000000:
                base_nlist = max(1000, int(base_nlist * 1.2))
        
            return IndexAndQueryParam(
                index_type='ivfpq',
                index_param={
                    'ivf_nlist': base_nlist,
                    'ivf_nlist_range': [1, 10000],
                    'num_subquantizers': 8,
                    'num_subquantizers_range': [1, 10000],
                },
                query_param={
                    'ivf_probes': 100,
                    'ivf_probes_range': [1, 10000],
                    'ivfpq_refine_k_factor': 2,
                    'ivfpq_refine_k_factor_range': [1, 64]
                }
            )
        
        # HNSW理论参数
        elif index_type == 'hnsw':            
            return IndexAndQueryParam(
                index_type='hnsw',
                index_param={
                    'm': 16,
                    'm_range': [8, 64],
                    'ef_construction': 200,
                    'ef_construction_range': [100, 800]
                },
                query_param={
                    'hnsw_ef_search': 100,
                    'hnsw_ef_search_range': [100, 2000]
                }
            )
        
        # DiskANN理论参数
        elif index_type == 'diskann':
            # 这里可以根据维度和数据量添加DiskANN的理论参数
            return IndexAndQueryParam(
                index_type='diskann',
                index_param={},
                query_param={
                    'diskann_search_list_size': 100,
                    'diskann_search_list_size_range': [1, 2000]
                }
            )
        
        else:
            raise ValueError(f"不支持的索引类型: {index_type}")