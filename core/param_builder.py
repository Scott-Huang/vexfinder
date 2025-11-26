#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Any
import math
from core.types import IndexAndQueryParam


def scale_parameters(index_type: str, params: Dict[str, Any], scale_factor: float) -> Dict[str, Any]:
    adjusted_params = params.copy()
    if index_type == "ivfflat":
        if "ivf_nlist" in adjusted_params:
            adjusted_params["ivf_nlist"] = max(4, min(10000, int(adjusted_params["ivf_nlist"] * math.sqrt(scale_factor))))
    elif index_type == "graph_index":
        if "ef_construction" in adjusted_params:
            adjusted_params["ef_construction"] = max(40, min(800, int(adjusted_params["ef_construction"] * (1 + 0.2 * math.log10(scale_factor)))))
    elif index_type == "ivfpq":
        if "ivf_nlist" in adjusted_params:
            adjusted_params["ivf_nlist"] = max(4, min(10000, int(adjusted_params["ivf_nlist"] * math.sqrt(scale_factor))))
    return adjusted_params

def scale_query_parameters(index_type: str, params: Dict[str, Any], scale_factor: float) -> Dict[str, Any]:
    adjusted_params = params.copy()
    if index_type == "ivfflat":
        if "ivf_probes" in adjusted_params:
            adjusted_params["ivf_probes"] = max(1, min(2000, int(adjusted_params["ivf_probes"] * (1 + 0.3 * math.log10(scale_factor)))))
    elif index_type == "graph_index":
        if "hnsw_ef_search" in adjusted_params:
            adjusted_params["hnsw_ef_search"] = max(adjusted_params["hnsw_ef_search"], 
                                                    min(2000, int(adjusted_params["hnsw_ef_search"] * (1 + 0.3 * math.log10(scale_factor)))))
    elif index_type == "ivfpq":
        if "ivf_probes" in adjusted_params:
            adjusted_params["ivf_probes"] = max(1, min(2000, int(adjusted_params["ivf_probes"] * (1 + 0.3 * math.log10(scale_factor)))))
        if "ivfpq_refine_k_factor" in adjusted_params:
            adjusted_params["ivfpq_refine_k_factor"] = max(1, min(64, int(adjusted_params["ivfpq_refine_k_factor"] * (1 + 0.2 * math.log10(scale_factor)))))
    elif index_type == "diskann":
        if "diskann_search_list_size" in adjusted_params:
            adjusted_params["diskann_search_list_size"] = max(adjusted_params["diskann_search_list_size"], 
                                                              min(2000, int(adjusted_params["diskann_search_list_size"] * (1 + 0.3 * math.log10(scale_factor)))))
    
    return adjusted_params

    
def get_theoretical_param(sample_table_count: int, index_type: str) -> IndexAndQueryParam:
    if not index_type:
        index_type = 'graph_index'
    if index_type == 'ivfflat':
        base_nlist = int(math.sqrt(sample_table_count))
        if sample_table_count < 10000:
            base_nlist = max(16, min(base_nlist, 64))
        elif sample_table_count > 1000000:
            base_nlist = max(1000, int(base_nlist * 1.2))
        nprobe_ratio = 0.05
        if sample_table_count < 10000:
            nprobe_ratio = 0.1
        elif sample_table_count > 1000000:
            nprobe_ratio = 0.03
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
    elif index_type == 'ivfpq':
        base_nlist = int(math.sqrt(sample_table_count))
        if sample_table_count < 10000:
            base_nlist = max(16, min(base_nlist, 64))
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
    elif index_type == 'graph_index': 
        return IndexAndQueryParam(
            index_type='graph_index',
            index_param={
                'm': 20,
                'm_range': [8, 64],
                'ef_construction': 200,
                'ef_construction_range': [100, 800],
                'quantizer': '\"none\"',
                'quantizer_range': ['\"none\"', 'pq', 'rabitq']
            },
            query_param={
                'hnsw_ef_search': 100,
                'hnsw_ef_search_range': [25, 2000]
            }
        )
    elif index_type == 'diskann':
        return IndexAndQueryParam(
            index_type='diskann',
            index_param={
                'm': 99,
                'm_range': [60, 200],
                'ef_construction': 200,
                'ef_construction_range': [100, 800],
                'occlusion_factor': 1.2,
                'occlusion_factor_range': [1.01, 1.2]
            },
            query_param={
                'diskann_search_list_size': 100,
                'diskann_search_list_size_range': [1, 2000]
            }
        )
    else:
        raise ValueError(f"不支持的索引类型: {index_type}")
