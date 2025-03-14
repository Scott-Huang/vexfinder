#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import importlib
from core.config import IndexConfig, PerformanceConfig
from core.engine import DatabaseEngine, db_engine
from core.module import MODULE_MAP, BaseModule
from core.types import QueryData, IndexAndQueryParam, TableInfoConfig
from core.recall import get_recall_values
from core.param_builder import IndexParamBuilder
from core.result_collector import ResultCollector
from typing import Dict, List, Any, Tuple, Optional
import copy
from core.logging import logger


def instantiate_module(module_name: str, *args, **kwargs) -> BaseModule:
    """
    创建一个 `BaseModule` 对象
    """
    module = importlib.import_module(f"core.module")
    constructor = getattr(module, MODULE_MAP[module_name])
    return constructor(*args, **kwargs)

def single_query(algo, v, count):
    """仅执行查询并返回原始结果"""
    start = time.time()
    distances = algo.query(v, count)
    query_time = time.time() - start
    return (query_time, distances)


def create_index(module):
    """Builds the ANN index."""
    start = time.time()
    module.create_index()
    create_index_time = time.time() - start
    logger.info(f"创建索引时间: {create_index_time:.2f} 秒")
    index_size = module.get_memory_usage()
    table_size = module.get_table_usage()
    logger.info(f"索引大小: {index_size/1024:0.2f} MB")
    logger.info(f"表大小: {table_size/1024:0.2f} MB")
    return create_index_time, index_size, table_size


class Analyzer:
    def __init__(self, index_config: IndexConfig, db_engine_obj: DatabaseEngine, query_data: list[QueryData], performance: PerformanceConfig, table_info: Optional[TableInfoConfig] = None):
        """
        Runner 类用于运行索引创建和查询。

        Args:
            index_config: 索引配置对象
            db_engine_obj: 数据库引擎对象
            query_data: 查询数据表，包含 id、向量、groundtruth的distance,
            limit: 查询数据表中查询的行数
            table_info: 表信息配置
        """
        self.index_config = index_config
        self.db_engine = db_engine_obj or db_engine
        self.query_data = query_data
        self.limit = performance.limit
        self.min_recall = performance.min_recall
        self.tolerance = performance.tolerance
        self.table_info = table_info
                # 创建结果收集器
        logger.info("创建结果收集器")
        self.result_collector = ResultCollector(index_config.find_index_type)

    def analyze(self):

        # 获取索引参数
        param_builder = IndexParamBuilder(self.table_info.sample_table_count, self.table_info.dimension, self.index_config.find_index_type)
        theoretical_params = param_builder.get_theoretical_param()
        logger.info(f"理论最优索引参数: {theoretical_params}")

        # 分析理论最优索引参数
        result = self.analyze_with_param(theoretical_params)
        self.result_collector.add_best_result(result)

        # 根据理论最优索引参数的返回结果，探测更多索引参数。比如查看创建索引的时间是否可以减少，查询时间是否可以减少，索引大小是否可以减少等。
        # 可以循环调用param_builder.recommend_new_params方法，根据上一次的结果来获取更多的索引参数。
        pass
        


        return result


    def analyze_with_param(self, param: IndexAndQueryParam):
        """
        分析索引和查询参数
        """
        index_param = param.index_param
        query_param = param.query_param
        if param.index_type != self.index_config.find_index_type:
            raise ValueError(f"索引类型不匹配: {param.index_type} != {self.index_config.find_index_type}")
        
        try:
            # 添加基本参数
            if not self.table_info:
                raise ValueError("表信息配置为空")
            full_params = {
                'table_name': self.table_info.sample_table_name,
                'vector_column_name': self.table_info.vector_column_name,
                'metric': self.table_info.metric,
                'db_engine_obj': self.db_engine
            }
            
            # 过滤掉索引参数中的range参数，只保留实际参数
            filtered_index_param = {k: v for k, v in index_param.items() if not k.endswith('_range')}
            
            # 合并过滤后的 index_param 到 full_params
            full_params.update(filtered_index_param)
            module = instantiate_module(self.index_config.find_index_type, **full_params)
            create_index_time, index_size, table_size = create_index(module)
        except Exception as e:
            raise ValueError(f"创建索引 {self.index_config.find_index_type} 时出错: {e}")
        
        # 获取参数策略
        param_info = self._get_param_strategy(self.index_config.find_index_type, query_param)
        if not param_info:
            raise ValueError(f"_get_param_strategy方法不支持自动调整 {self.index_config.find_index_type} 索引类型的参数")
    
        analyze_result = {
            "create_index_time": create_index_time,
            "index_size": index_size,
            "table_size": table_size,
            "index_type": param.index_type,
            "index_param": index_param,
            "query_param": query_param,
            "user_min_recall": self.min_recall,
            "user_tolerance": self.tolerance,
            "success": False
        }

        # 预热缓存
        if self.index_config.prepare_cache:
            # 获取第一组参数进行预热
            module.set_query_arguments(**query_param)
            # 请所有的请求请求一遍, 预热图索引
            for query_data in self.query_data:
                module.query(query_data.vectors, self.limit)

        # 测试查询参数
        recall, avg_query_time, min_query_time, max_query_time, qps = self.test_query_param(module, query_param)

        # 记录初始参数和性能
        best_param = copy.deepcopy(query_param)
        best_performance = {
            "recall": recall,
            "avg_query_time": avg_query_time,
            "min_query_time": min_query_time,
            "max_query_time": max_query_time,
            "qps": qps
        }
        
        # 如果当前召回率低于目标值，增加参数值
        if recall + self.tolerance < self.min_recall:
            logger.info(f"参数组合的召回率 {recall:.6f} 低于最小召回率 {self.min_recall:.6f}，探索更大的参数")
            best_param, best_performance = self._increase_param_until_target(
                module, query_param, param_info)
        
        # 如果当前召回率高于目标值很多（接近1或超过min_recall + 0.1），减小参数值
        elif recall > min(self.min_recall + 0.1, 0.9999):  # 使用0.1作为固定阈值，避免陷入循环
            logger.info(f"参数组合的召回率 {recall:.6f} 远高于最小召回率 {self.min_recall:.6f}，探索更小的参数")
            best_param, best_performance = self._decrease_param_until_target(
                module, query_param, param_info)
            
        logger.info(f"最佳参数: {best_param}")
        logger.info(f"最佳性能: 召回率={best_performance['recall']:.6f}, " 
              f"平均查询时间={best_performance['avg_query_time']:.6f}秒, "
              f"QPS={best_performance['qps']:.4f}")
        
        # 更新描述信息并返回结果
        analyze_result.update({
            "best_query_param": best_param,
            "best_performance": best_performance,
            "success": bool(best_performance["recall"] + self.tolerance >= self.min_recall)
        })
        
        return analyze_result
        
    def _get_param_strategy(self, index_type: str, query_param: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据索引类型返回关键参数信息
        
        Args:
            index_type: 索引类型
            query_param: 查询参数字典，用于获取参数范围
            
        Returns:
            Dict: 包含参数名、范围和基础步长的字典
        """
        result = {}
        
        if index_type == "ivfflat":
            result = {
                "primary_param": {
                    "name": "ivf_probes",
                    "base_step": 5,
                    "min_value": query_param.get("ivf_probes_range", [1, 10000])[0],
                    "max_value": query_param.get("ivf_probes_range", [1, 10000])[1]
                }
            }
        elif index_type == "ivfpq":
            # IVFPQ同时需要调整两个参数
            result = {
                "primary_param": {
                    "name": "ivf_probes",
                    "base_step": 5,
                    "min_value": query_param.get("ivf_probes_range", [1, 10000])[0],
                    "max_value": query_param.get("ivf_probes_range", [1, 10000])[1]
                },
                "secondary_param": {
                    "name": "ivfpq_refine_k_factor",
                    "base_step": 1,
                    "min_value": query_param.get("ivfpq_refine_k_factor_range", [1, 64])[0],
                    "max_value": query_param.get("ivfpq_refine_k_factor_range", [1, 64])[1]
                }
            }
        elif index_type == "hnsw":
            result = {
                "primary_param": {
                    "name": "hnsw_ef_search",
                    "base_step": 10,
                    "min_value": query_param.get("hnsw_ef_search_range", [1, 2000])[0],
                    "max_value": query_param.get("hnsw_ef_search_range", [1, 2000])[1]
                }
            }
        elif index_type == "diskann":
            result = {
                "primary_param": {
                    "name": "diskann_search_list_size",
                    "base_step": 5,
                    "min_value": query_param.get("diskann_search_list_size_range", [1, 2000])[0],
                    "max_value": query_param.get("diskann_search_list_size_range", [1, 2000])[1]
                }
            }
            
        return result
    
    def _calculate_step(self, current_param_value: int, current_recall: float, min_recall: float, base_step: int) -> int:
        """
        根据召回率差值计算动态步长
        
        Args:
            current_param_value: 当前参数值
            current_recall: 当前召回率
            min_recall: 目标召回率
            base_step: 基础步长
            
        Returns:
            int: 计算出的步长
        """
        if current_recall < min_recall:
            # 计算召回率差值
            recall_diff = min_recall - current_recall
            # 根据差值比例调整步长，差距越大步长越大
            step_factor = max(0.1, min(1.0, recall_diff / 0.1))  # 将差值归一化到0.1-1.0范围
            adjusted_step = max(base_step, int(current_param_value * step_factor))
            return adjusted_step
        else:
            # 如果已经达到目标召回率，使用更小的步长进行微调
            return max(1, base_step // 2)
            
    def _increase_param_until_target(self, module, query_param, param_info):
        """
        增加参数值直到达到目标召回率
        
        Args:
            module: 索引模块
            query_param: 查询参数字典
            param_info: 参数信息字典，包含主参数和可能的次要参数信息
            
        Returns:
            (最佳参数, 最佳性能)元组
        """
        current_param = copy.deepcopy(query_param)
        best_param = copy.deepcopy(query_param)
        
        # 获取主参数信息
        primary_param = param_info["primary_param"]
        primary_name = primary_param["name"]
        primary_base_step = primary_param["base_step"]
        primary_min = primary_param["min_value"]
        primary_max = primary_param["max_value"]
        
        # 获取次要参数信息（如果有）
        has_secondary = "secondary_param" in param_info
        if has_secondary:
            secondary_param = param_info["secondary_param"]
            secondary_name = secondary_param["name"]
            secondary_base_step = secondary_param["base_step"]
            secondary_min = secondary_param["min_value"]
            secondary_max = secondary_param["max_value"]
        
        # 测试初始参数，作为基准
        initial_recall, initial_avg_query_time, initial_min_query_time, initial_max_query_time, initial_qps = self.test_query_param(
            module, current_param)
        
        best_performance = {
            "recall": initial_recall,
            "avg_query_time": initial_avg_query_time,
            "min_query_time": initial_min_query_time,
            "max_query_time": initial_max_query_time,
            "qps": initial_qps
        }
        
        if has_secondary:
            logger.info(f"初始参数 {primary_name}={current_param[primary_name]}, {secondary_name}={current_param[secondary_name]}, 召回率={initial_recall:.6f}, 查询时间={initial_avg_query_time:.6f}秒, QPS={initial_qps:.4f}")
        else:
            logger.info(f"初始参数 {primary_name}={current_param[primary_name]}, 召回率={initial_recall:.6f}, 查询时间={initial_avg_query_time:.6f}秒, QPS={initial_qps:.4f}")
        
        max_iterations = 20  # 最大迭代次数
        iterations = 0
        
        # 尝试增加参数策略：先尝试只增加主参数，如果效果不明显再同时增加次要参数
        primary_only_phase = True
        secondary_adjusted = False
        
        while iterations < max_iterations:
            iterations += 1
            
            # 根据当前召回率和目标召回率动态计算主参数步长
            primary_step = self._calculate_step(
                current_param[primary_name], 
                best_performance["recall"], 
                self.min_recall,
                primary_base_step
            )
            
            # 计算新的主参数值，确保在允许范围内
            new_primary_value = min(primary_max, current_param[primary_name] + primary_step)
            
            # 如果主参数已经达到最大值且召回率仍不满足，且有次要参数，则开始调整次要参数
            if (new_primary_value == current_param[primary_name] == primary_max and 
                best_performance["recall"] < self.min_recall and 
                has_secondary and primary_only_phase):
                logger.info(f"主参数 {primary_name} 已达最大值 {primary_max}，开始调整次要参数 {secondary_name}")
                primary_only_phase = False
                
            # 更新主参数值
            current_param[primary_name] = new_primary_value
            
            # 如果不再只调整主参数，同时调整次要参数（如果有）
            if has_secondary and not primary_only_phase:
                secondary_step = self._calculate_step(
                    current_param[secondary_name], 
                    best_performance["recall"], 
                    self.min_recall,
                    secondary_base_step
                )
                
                # 计算新的次要参数值，确保在允许范围内
                new_secondary_value = min(secondary_max, current_param[secondary_name] + secondary_step)
                current_param[secondary_name] = new_secondary_value
                secondary_adjusted = True
                
                logger.info(f"使用动态步长，将 {primary_name} 增加到 {new_primary_value}，{secondary_name} 增加到 {new_secondary_value}")
            else:
                logger.info(f"使用动态步长 {primary_step}，将 {primary_name} 增加到 {new_primary_value}")
            
            # 如果参数值没有变化，说明已经达到了参数上限
            if (current_param[primary_name] == best_param[primary_name] and 
                (not has_secondary or current_param[secondary_name] == best_param[secondary_name])):
                logger.info(f"参数已达上限，无法继续增加")
                break
            
            # 测试新参数
            recall, avg_query_time, min_query_time, max_query_time, qps = self.test_query_param(
                module, current_param)
            
            current_performance = {
                "recall": recall,
                "avg_query_time": avg_query_time,
                "min_query_time": min_query_time,
                "max_query_time": max_query_time,
                "qps": qps
            }
            
            if has_secondary:
                logger.info(f"参数 {primary_name}={current_param[primary_name]}, {secondary_name}={current_param[secondary_name]}, 召回率={recall:.6f}, 查询时间={avg_query_time:.6f}秒, QPS={qps:.4f}")
            else:
                logger.info(f"参数 {primary_name}={current_param[primary_name]}, 召回率={recall:.6f}, 查询时间={avg_query_time:.6f}秒, QPS={qps:.4f}")
            
            # 如果达到目标召回率，记录并退出
            if recall >= self.min_recall:
                best_param = copy.deepcopy(current_param)
                best_performance = current_performance
                logger.info(f"找到满足目标召回率的参数组合, 召回率={recall:.6f}")
                break
            
            # 如果不断增加参数，召回率增长不明显（小于1%），可能已接近上限
            if recall - best_performance["recall"] < 0.01:
                logger.info(f"召回率增长缓慢（小于1%），可能已接近上限")
                
                # 如果是IVFPQ且只调整了主参数，尝试开始调整次要参数
                if has_secondary and primary_only_phase:
                    logger.info(f"尝试调整次要参数 {secondary_name} 以进一步提高召回率")
                    primary_only_phase = False
                    continue
            
            # 更新最佳性能记录（如果当前更好）
            if recall > best_performance["recall"]:
                best_param = copy.deepcopy(current_param)
                best_performance = current_performance
        
        # 如果没有找到满足条件的参数，使用最后记录的最佳参数
        if best_performance["recall"] < self.min_recall:
            logger.info(f"警告：在{max_iterations}次迭代后仍未找到满足目标召回率 {self.min_recall} 的参数")
            logger.info(f"最佳找到的召回率为 {best_performance['recall']:.6f}")
            
            # 如果是IVFPQ且还没有调整过次要参数，尝试调整
            if has_secondary and not secondary_adjusted:
                logger.info(f"尝试调整次要参数 {secondary_name} 以进一步提高召回率")
                # 恢复到最佳参数，然后尝试调整次要参数
                current_param = copy.deepcopy(best_param)
                
                # 尝试几次增加次要参数
                for i in range(5):
                    # 增加次要参数值
                    secondary_step = max(1, secondary_base_step * (i + 1))
                    new_secondary_value = min(secondary_max, current_param[secondary_name] + secondary_step)
                    
                    if new_secondary_value == current_param[secondary_name]:
                        break  # 已达最大值
                        
                    current_param[secondary_name] = new_secondary_value
                    logger.info(f"增加次要参数 {secondary_name} 到 {new_secondary_value}")
                    
                    # 测试新参数
                    recall, avg_query_time, min_query_time, max_query_time, qps = self.test_query_param(
                        module, current_param)
                    
                    logger.info(f"参数 {primary_name}={current_param[primary_name]}, {secondary_name}={current_param[secondary_name]}, 召回率={recall:.6f}, 查询时间={avg_query_time:.6f}秒, QPS={qps:.4f}")
                    
                    # 如果有改进，更新最佳参数
                    if recall > best_performance["recall"]:
                        best_param = copy.deepcopy(current_param)
                        best_performance = {
                            "recall": recall,
                            "avg_query_time": avg_query_time,
                            "min_query_time": min_query_time,
                            "max_query_time": max_query_time,
                            "qps": qps
                        }
                        logger.info(f"次要参数调整改进了召回率，新召回率={recall:.6f}")
                        
                        # 如果达到目标，退出
                        if recall >= self.min_recall:
                            logger.info(f"找到满足目标召回率的参数组合")
                            break
        
        return best_param, best_performance
    
    def _decrease_param_until_target(self, module, query_param, param_info):
        """
        减小参数值直到低于目标召回率，然后回退一步
        
        Args:
            module: 索引模块
            query_param: 查询参数字典
            param_info: 参数信息字典，包含主参数和可能的次要参数信息
            
        Returns:
            (最佳参数, 最佳性能)元组
        """
        current_param = copy.deepcopy(query_param)
        best_param = copy.deepcopy(query_param)
        
        # 获取主参数信息
        primary_param = param_info["primary_param"]
        primary_name = primary_param["name"]
        primary_base_step = primary_param["base_step"]
        primary_min = primary_param["min_value"]
        primary_max = primary_param["max_value"]
        
        # 获取次要参数信息（如果有）
        has_secondary = "secondary_param" in param_info
        if has_secondary:
            secondary_param = param_info["secondary_param"]
            secondary_name = secondary_param["name"]
            secondary_base_step = secondary_param["base_step"]
            secondary_min = secondary_param["min_value"]
            secondary_max = secondary_param["max_value"]
        
        # 先测试当前参数
        recall, avg_query_time, min_query_time, max_query_time, qps = self.test_query_param(
            module, current_param)
        
        best_performance = {
            "recall": recall,
            "avg_query_time": avg_query_time,
            "min_query_time": min_query_time,
            "max_query_time": max_query_time,
            "qps": qps
        }
        
        if has_secondary:
            logger.info(f"初始参数 {primary_name}={current_param[primary_name]}, {secondary_name}={current_param[secondary_name]}, 召回率={recall:.6f}, 查询时间={avg_query_time:.6f}秒, QPS={qps:.4f}")
        else:
            logger.info(f"初始参数 {primary_name}={current_param[primary_name]}, 召回率={recall:.6f}, 查询时间={avg_query_time:.6f}秒, QPS={qps:.4f}")
        
        max_iterations = 20  # 最大迭代次数
        iterations = 0
        
        # 减小参数策略：先尝试减小次要参数（如果有），然后再减小主参数
        secondary_phase = has_secondary
        
        while iterations < max_iterations:
            iterations += 1
            
            if secondary_phase:
                # 计算次要参数的减小步长
                secondary_step = self._calculate_step(
                    current_param[secondary_name], 
                    best_performance["recall"], 
                    self.min_recall,
                    secondary_base_step
                ) // 2  # 减小步长应小于增加步长
                
                # 确保步长最小为1
                secondary_step = max(1, secondary_step)
                
                # 计算新的次要参数值，确保在允许范围内
                new_secondary_value = max(secondary_min, current_param[secondary_name] - secondary_step)
                
                # 如果次要参数已经达到最小值，开始减小主参数
                if new_secondary_value == current_param[secondary_name] == secondary_min:
                    logger.info(f"次要参数 {secondary_name} 已达最小值 {secondary_min}，开始调整主参数 {primary_name}")
                    secondary_phase = False
                    continue
                
                # 更新次要参数值
                current_param[secondary_name] = new_secondary_value
                
                logger.info(f"减小次要参数 {secondary_name} 到 {new_secondary_value}")
            else:
                # 计算主参数的减小步长
                primary_step = self._calculate_step(
                    current_param[primary_name], 
                    best_performance["recall"], 
                    self.min_recall,
                    primary_base_step
                ) // 2  # 减小步长应小于增加步长
                
                # 确保步长最小为1
                primary_step = max(1, primary_step)
                
                # 计算新的主参数值，确保在允许范围内
                new_primary_value = max(primary_min, current_param[primary_name] - primary_step)
                
                # 如果主参数已经达到最小值，退出循环
                if new_primary_value == current_param[primary_name] == primary_min:
                    logger.info(f"主参数 {primary_name} 已达最小值 {primary_min}，无法继续减小")
                    break
                
                # 更新主参数值
                current_param[primary_name] = new_primary_value
                
                logger.info(f"减小主参数 {primary_name} 到 {new_primary_value}")
            
            # 如果参数值没有变化，继续下一次迭代
            if (not secondary_phase and current_param[primary_name] == best_param[primary_name]) or \
               (secondary_phase and current_param[secondary_name] == best_param[secondary_name]):
                continue
            
            # 测试新参数
            recall, avg_query_time, min_query_time, max_query_time, qps = self.test_query_param(
                module, current_param)
            
            current_performance = {
                "recall": recall,
                "avg_query_time": avg_query_time,
                "min_query_time": min_query_time,
                "max_query_time": max_query_time,
                "qps": qps
            }
            
            if has_secondary:
                logger.info(f"参数 {primary_name}={current_param[primary_name]}, {secondary_name}={current_param[secondary_name]}, 召回率={recall:.6f}, 查询时间={avg_query_time:.6f}秒, QPS={qps:.4f}")
            else:
                logger.info(f"参数 {primary_name}={current_param[primary_name]}, 召回率={recall:.6f}, 查询时间={avg_query_time:.6f}秒, QPS={qps:.4f}")
            
            # 如果召回率仍然满足条件，更新最佳参数
            if recall >= self.min_recall:
                best_param = copy.deepcopy(current_param)
                best_performance = current_performance
                logger.info(f"找到更优的参数组合, 召回率仍满足要求={recall:.6f}")
            else:
                # 如果召回率低于目标，记录最后一个满足条件的参数
                logger.info(f"召回率 {recall:.6f} 低于目标值 {self.min_recall}，回到上一个参数组合")
                
                # 如果是首次迭代就低于目标，可能需要尝试其他策略
                if iterations == 1:
                    # 如果是次要参数阶段，尝试减小主参数
                    if secondary_phase:
                        # 恢复次要参数，转到主参数阶段
                        current_param[secondary_name] = best_param[secondary_name]
                        secondary_phase = False
                        continue
                
                break  # 否则停止迭代
        
        logger.info(f"减小参数搜索结束，选择的最佳参数组合的召回率={best_performance['recall']:.6f}")
        return best_param, best_performance

    def test_query_param(self, module, query_param: Dict[str, Any]):
        """
        执行查询
        """
        # 过滤掉查询参数中的range参数，只保留实际参数
        filtered_query_param = {k: v for k, v in query_param.items() if not k.endswith('_range')}
        
        module.set_query_arguments(**filtered_query_param)
        
        results = []
        for query_data in self.query_data:
            results.append(single_query(module, query_data.vectors, self.limit))

        #计算召回率
        recall = get_recall_values([query_data.distances for query_data in self.query_data], [distances for _, distances in results], self.limit)
        
        # 计算平均查询时间
        avg_query_time = sum(time for time, _ in results) / len(results)

        # 最小查询时间
        min_query_time = min(time for time, _ in results)

        # 最大查询时间
        max_query_time = max(time for time, _ in results)

        # 计算QPS
        qps = 1 / avg_query_time

        return recall, avg_query_time, min_query_time, max_query_time, qps



