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
        dimension = self.dimension
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

    def recommend_new_params(self, current_params: Dict[str, Any], direction: Literal['left', 'right'] = 'right') -> Dict[str, Any]:
        """
        根据当前参数推荐新的参数
        
        Args:
            current_params: 当前索引参数
            direction: 探索方向，'left'表示更注重性能，'right'表示更注重精度
            
        Returns:
            推荐的新参数
        """
        # 克隆当前参数以避免修改原始对象
        new_params = current_params.copy()
        
        # 获取索引类型
        index_type = current_params.get('index_type') or self.index_type
        if not index_type:
            # 尝试从参数名推断索引类型
            if 'ivf_nlist' in current_params:
                index_type = 'ivfflat'
            elif 'M' in current_params and 'efConstruction' in current_params:
                index_type = 'hnsw'
            else:
                # 如果无法推断，使用类成员变量中的索引类型
                index_type = self.index_type
        
        # 获取数据维度和样本数量，用于更智能地调整参数
        dimension = self.dimension
        sample_count = self.sample_table_count
        
        # 系数：向左时和向右时的调整因子
        # 对于大数据集和高维数据，使用更温和的调整因子
        if sample_count > 1000000 or dimension > 256:
            left_factor = 0.85 if direction == 'left' else 1.0
            right_factor = 1.0 if direction == 'left' else 1.15
        else:
            left_factor = 0.8 if direction == 'left' else 1.0
            right_factor = 1.0 if direction == 'left' else 1.25
        
        if index_type == 'ivfflat':
            # 处理IVF-FLAT参数
            if 'ivf_nlist' in current_params:
                nlist = current_params['ivf_nlist']
                if direction == 'left':
                    # 向左：增加nlist，减少nprobe（注重性能）
                    # 但对于小数据集，不要增加太多
                    if sample_count < 10000:
                        new_params['nlist'] = int(nlist * 1.15)
                    else:
                        new_params['nlist'] = int(nlist * (1/left_factor))
                else:
                    # 向右：减少nlist，增加nprobe（注重精度）
                    # 但保证nlist不会太小
                    min_nlist = 4 if sample_count < 10000 else 16
                    new_params['nlist'] = max(min_nlist, int(nlist * right_factor))
            
            if 'ivf_probes' in current_params:
                nprobe = current_params['ivf_probes']
                if direction == 'left':
                    # 向左：减少nprobe（注重性能）
                    # 但确保nprobe不要太小
                    new_params['ivf_probes'] = max(1, int(nprobe * left_factor))
                else:
                    # 向右：增加nprobe（注重精度）
                    # 对于高维数据，可能需要更大的nprobe
                    if dimension > 128:
                        new_params['ivf_probes'] = int(nprobe * (1/right_factor) * 1.1)
                    else:
                        new_params['ivf_probes'] = int(nprobe * (1/right_factor))
        
        elif index_type == 'ivfpq':
            # 处理IVF-PQ参数
            if 'nlist' in current_params:
                nlist = current_params['nlist']
                if direction == 'left':
                    # 向左：增加nlist（注重性能）
                    new_params['nlist'] = int(nlist * (1/left_factor))
                else:
                    # 向右：减少nlist（注重精度）
                    min_nlist = 4 if sample_count < 10000 else 16
                    new_params['nlist'] = max(min_nlist, int(nlist * right_factor))
            
            if 'nprobe' in current_params:
                nprobe = current_params['nprobe']
                if direction == 'left':
                    # 向左：减少nprobe（注重性能）
                    new_params['nprobe'] = max(1, int(nprobe * left_factor))
                else:
                    # 向右：增加nprobe（注重精度）
                    new_params['nprobe'] = int(nprobe * (1/right_factor))
            
            if 'm' in current_params:
                m = current_params['m']
                if direction == 'left':
                    # 向左：减少m（注重性能）
                    # 但m需要是维度的因子
                    candidates = [8, 16, 24, 32]
                    # 选择小于当前m的最大因子
                    new_m = max([c for c in candidates if c < m], default=max(8, m - 8))
                    # 检查是否为维度的因子
                    if dimension % new_m == 0:
                        new_params['m'] = new_m
                    else:
                        new_params['m'] = max(8, m - 8)
                else:
                    # 向右：增加m（注重精度）
                    # 但m不应超过维度，且优先选择维度的因子
                    candidates = [8, 16, 24, 32, 48, 64, 96]
                    # 选择大于当前m的最小因子，但不超过维度
                    potential_m = min([c for c in candidates if c > m and c < dimension], default=min(96, m + 8))
                    # 检查是否为维度的因子
                    if dimension % potential_m == 0:
                        new_params['m'] = potential_m
                    else:
                        new_params['m'] = min(96, m + 8)
            
            if 'refine_factor' in current_params:
                refine_factor = current_params['refine_factor']
                if direction == 'left':
                    # 向左：减少refine_factor（注重性能）
                    new_params['refine_factor'] = max(1, int(refine_factor * left_factor))
                else:
                    # 向右：增加refine_factor（注重精度）
                    # 高维数据可能需要更大的refine_factor
                    if dimension > 128:
                        new_params['refine_factor'] = int(refine_factor * (1/right_factor) * 1.1) + 1
                    else:
                        new_params['refine_factor'] = int(refine_factor * (1/right_factor)) + 1
        
        elif index_type == 'hnsw':
            # 处理HNSW参数
            if 'M' in current_params:
                M = current_params['M']
                if direction == 'left':
                    # 向左：减少M（注重性能）
                    # 但对于高维数据，M不应该太小
                    min_M = 8 if dimension <= 64 else 12
                    new_params['M'] = max(min_M, int(M * left_factor))
                else:
                    # 向右：增加M（注重精度）
                    # 但M不应该太大，尤其是对于低维数据
                    max_M = 48 if dimension <= 64 else 64
                    new_params['M'] = min(max_M, int(M * (1/right_factor)))
            
            if 'efConstruction' in current_params and 'M' in new_params:
                efConstruction = current_params['efConstruction']
                M = new_params['M']  # 使用可能已经调整过的M值
                if direction == 'left':
                    # 向左：减少efConstruction（注重性能）
                    # 但efConstruction应该至少是M的倍数
                    new_params['efConstruction'] = max(M * 2, int(efConstruction * left_factor))
                else:
                    # 向右：增加efConstruction（注重精度）
                    # 对于大数据集，可能需要更大的efConstruction
                    if sample_count > 100000:
                        new_params['efConstruction'] = int(efConstruction * (1/right_factor) * 1.1)
                    else:
                        new_params['efConstruction'] = int(efConstruction * (1/right_factor))
            
            if 'ef' in current_params:
                ef = current_params['ef']
                if direction == 'left':
                    # 向左：减少ef（注重性能）
                    # 但ef应该至少是M的倍数
                    M = new_params.get('M', current_params.get('M', 16))
                    new_params['ef'] = max(M, int(ef * left_factor))
                else:
                    # 向右：增加ef（注重精度）
                    # 对于高维数据，可能需要更大的ef
                    if dimension > 128:
                        new_params['ef'] = int(ef * (1/right_factor) * 1.1)
                    else:
                        new_params['ef'] = int(ef * (1/right_factor))
        
        elif index_type == 'diskann':
            # 处理DiskANN参数 - 可以根据维度和数据量进行更智能的调整
            # 这里根据DiskANN的具体参数进行调整
            pass
        
        logger.info(f"推荐新参数：从{'性能' if direction=='left' else '精度'}方向对{index_type}进行调整")
        return new_params
    
    def get_params_pair(self, index_type: str = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        获取指定索引类型的理论最优索引参数和查询参数对
        
        Args:
            index_type: 索引类型，例如'ivfflat', 'ivfpq', 'hnsw'等。如果为None，则使用实例的索引类型。
            
        Returns:
            (索引参数, 查询参数)元组
        """
        # 如果指定了索引类型，暂时保存当前索引类型并设置新的索引类型
        original_index_type = self.index_type
        if index_type is not None:
            self.index_type = index_type
        
        try:
            # 获取理论参数
            params = self.get_theoretical_params()
            # 返回索引参数和查询参数对
            return (params['index_params'], params['query_params'])
        finally:
            # 恢复原始索引类型
            if index_type is not None:
                self.index_type = original_index_type
