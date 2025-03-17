#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Any
import math
from core.types import IndexAndQueryParam


def scale_parameters(index_type: str, params: Dict[str, Any], scale_factor: float) -> Dict[str, Any]:
    """
    根据数据规模调整参数
    
    Args:
        index_type: 索引类型
        params: 原始参数
        scale_factor: 缩放因子(源数据大小/采样数据大小)
        
    Returns:
        Dict: 调整后的参数
    """
    adjusted_params = params.copy()
    
    # 根据索引类型进行参数调整
    if index_type == "ivfflat":
        # IVF索引的聚类中心数与数据规模有关，大约与数据量的平方根成正比
        if "ivf_nlist" in adjusted_params:
            # 使用平方根比例缩放nlist
            adjusted_params["ivf_nlist"] = max(4, min(10000, int(adjusted_params["ivf_nlist"] * math.sqrt(scale_factor))))
    
    elif index_type == "hnsw":
        # HNSW索引的参数通常与数据规模关系不大，但可能需要微调
        # ef_construction可能需要随数据量增加而略微增加
        if "ef_construction" in adjusted_params:
            # 使用对数缩放ef_construction
            adjusted_params["ef_construction"] = max(40, min(800, int(adjusted_params["ef_construction"] * (1 + 0.2 * math.log10(scale_factor)))))
    
    elif index_type == "ivfpq":
        # IVF-PQ索引的聚类中心数与数据规模有关
        if "ivf_nlist" in adjusted_params:
            adjusted_params["ivf_nlist"] = max(4, min(10000, int(adjusted_params["ivf_nlist"] * math.sqrt(scale_factor))))
        # num_subquantizers通常不需要随数据量变化
    
    
    return adjusted_params

def scale_query_parameters(index_type: str, params: Dict[str, Any], scale_factor: float) -> Dict[str, Any]:
    """
    根据数据规模调整查询参数
    
    Args:
        index_type: 索引类型
        params: 原始查询参数
        scale_factor: 缩放因子(源数据大小/采样数据大小)
        
    Returns:
        Dict: 调整后的查询参数
    """
    adjusted_params = params.copy()
    
    # 根据索引类型进行查询参数调整
    if index_type == "ivfflat":
        # 查询时探测的聚类中心数可能需要增加
        if "ivf_probes" in adjusted_params:
            # 使用对数缩放probes
            adjusted_params["ivf_probes"] = max(1, min(2000, int(adjusted_params["ivf_probes"] * (1 + 0.3 * math.log10(scale_factor)))))
    
    elif index_type == "hnsw":
        # 增加ef_search以保持召回率
        if "hnsw_ef_search" in adjusted_params:
            # 使用对数缩放ef_search
            adjusted_params["hnsw_ef_search"] = max(adjusted_params["hnsw_ef_search"], 
                                                    min(2000, int(adjusted_params["hnsw_ef_search"] * (1 + 0.3 * math.log10(scale_factor)))))
    
    elif index_type == "ivfpq":
        # 增加探测的聚类中心数
        if "ivf_probes" in adjusted_params:
            adjusted_params["ivf_probes"] = max(1, min(2000, int(adjusted_params["ivf_probes"] * (1 + 0.3 * math.log10(scale_factor)))))
        
        # 可能需要增加ivfpq_refine_k_factor以保持召回率
        if "ivfpq_refine_k_factor" in adjusted_params:
            adjusted_params["ivfpq_refine_k_factor"] = max(1, min(64, int(adjusted_params["ivfpq_refine_k_factor"] * (1 + 0.2 * math.log10(scale_factor)))))
    
    elif index_type == "diskann":
        # 增加search_list_size以保持召回率
        if "diskann_search_list_size" in adjusted_params:
            adjusted_params["diskann_search_list_size"] = max(adjusted_params["diskann_search_list_size"], 
                                                            min(2000, int(adjusted_params["diskann_search_list_size"] * (1 + 0.3 * math.log10(scale_factor)))))
    
    return adjusted_params

    
def get_theoretical_param(sample_table_count: int, index_type: str) -> IndexAndQueryParam:
    """
    获取理论最优的索引参数和查询参数
    
    根据数据维度和行数计算理论最优参数。
    根据index_type返回对应的参数。
    
    Returns:
        IndexParam对象
    """
    
    # 如果未指定索引类型，默认使用ivfflat
    if not index_type:
        index_type = 'ivfflat'
    
    # IVF-FLAT理论参数
    if index_type == 'ivfflat':
        # 根据数据量调整nlist值
        base_nlist = int(math.sqrt(sample_table_count))
        
        # 小数据集，使用较小的nlist以获得更好的精度
        if sample_table_count < 10000:
            base_nlist = max(16, min(base_nlist, 64))
        # 大数据集，使用略大的nlist以提高性能
        elif sample_table_count > 1000000:
            base_nlist = max(1000, int(base_nlist * 1.2))
        
        # 根据数据量调整nprobe比例
        nprobe_ratio = 0.05  # 默认为5%
        if sample_table_count < 10000:
            nprobe_ratio = 0.1  # 小数据集使用更大的比例
        elif sample_table_count > 1000000:
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
        base_nlist = int(math.sqrt(sample_table_count))
        
        # 小数据集，使用较小的nlist以获得更好的精度
        if sample_table_count < 10000:
            base_nlist = max(16, min(base_nlist, 64))
        # 大数据集，使用略大的nlist以提高性能
        elif sample_table_count > 1000000:
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