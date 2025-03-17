#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import importlib
import os
import datetime
from core.config import Config
from core.engine import DatabaseEngine, db_engine
from core.module import MODULE_MAP, BaseModule
from core.types import QueryData, IndexAndQueryParam 
from core.recall import get_recall_values
from core.param_builder import IndexParamBuilder
from typing import Dict, Any
import copy
from core.logging import logger

# 导入optuna相关模块
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances


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
    def __init__(self, db_engine_obj: DatabaseEngine, query_data: list[QueryData],  config_obj: Config):
        """
        Runner 类用于运行索引创建和查询。

        Args:
            index_config: 索引配置对象
            db_engine_obj: 数据库引擎对象
            query_data: 查询数据表，包含 id、向量、groundtruth的distance,
            limit: 查询数据表中查询的行数
            table_info: 表信息配置
        """
        self.config = config_obj
        self.index_config = config_obj.index_config
        self.db_engine = db_engine_obj or db_engine
        self.query_data = query_data
        self.performance = config_obj.performance
        self.min_recall = config_obj.performance.min_recall
        self.table_info = config_obj.table_info
        self.parallel_workers = config_obj.parallel_workers
        self.initial_explore_params = config_obj.initial_explore_params

        # 创建结果存储目录
        if not os.path.isdir(os.path.join("results")):
            os.makedirs(os.path.join("results"))

        # 创建索引类型_日期时间格式的文件夹
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.result_dir = os.path.join("results", f"{self.index_config.find_index_type}_{timestamp}")
        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir)


    def analyze(self):
        """
        使用Optuna优化索引创建参数，让analyze_with_param自动找到满足召回率要求的查询参数
        """
        # 获取理论最优参数作为起点
        if self.initial_explore_params.manual_param:
            if not self.initial_explore_params.index_param or not self.initial_explore_params.query_param:
                raise ValueError("当前initial_explore_params配置的 manual_param 为true，索引参数和查询参数不能为空")
            theoretical_params = {
                "index_param": self.initial_explore_params.index_param,
                "query_param": self.initial_explore_params.query_param
            }
            logger.info(f"使用用户设置参数作为优化起点: {theoretical_params}")
        else:
            param_builder = IndexParamBuilder(self.table_info.sample_table_count, 
                                          self.table_info.dimension, 
                                          self.index_config.find_index_type)
            theoretical_params = param_builder.get_theoretical_param()
            logger.info(f"获取理论最优参数作为优化起点: {theoretical_params}")
        
        # 检查理论参数中index_param是否为空
        is_index_param_empty = not theoretical_params.index_param or all(
            k.endswith('_range') for k in theoretical_params.index_param.keys()
        )
        
        # 如果索引参数为空，则直接使用理论参数调用analyze_with_param
        if is_index_param_empty:
            logger.info("索引参数为空，仅需搜索查询参数，跳过索引参数优化过程")
            # 获取结果
            result = self.analyze_with_param(theoretical_params)
            return result
            
        # 索引参数范围定义 (根据不同索引类型)
        index_param_ranges = {
            "ivfflat": {
                "ivf_nlist": (10, 2000),  # IVF聚类中心数量
            },
            "hnsw": {
                "ef_construction": (40, 800),  # 建图时的候选邻居数
                "m": (8, 96),  # 每个节点的边数
            },
            "ivfpq": {
                "ivf_nlist": (10, 2000),  # IVF聚类中心数量
                "num_subquantizers": (1, 64),  # PQ编码的子向量数量
            },
            "diskann": {}
        }
        
        # 获取向量维度，用于IVFPQ参数约束
        vector_dimension = self.table_info.dimension
        logger.info(f"向量维度: {vector_dimension}")
        
        # 如果是IVFPQ索引类型，计算合法的num_subquantizers值列表
        # num_subquantizers需要满足两个条件：
        # 1. 小于原始向量维度
        # 2. 原始向量维度必须是num_subquantizers的整数倍
        if self.index_config.find_index_type == "ivfpq":
            valid_subquantizers = []
            for i in range(1, min(64, vector_dimension)):
                if vector_dimension % i == 0:
                    valid_subquantizers.append(i)
            
            if not valid_subquantizers:
                logger.warning(f"无法找到合适的num_subquantizers值，向量维度: {vector_dimension}")
                # 如果找不到合法值，使用一些常用值作为候选，但后续检查是否可用
                valid_subquantizers = [1]
            
            logger.info(f"合法的num_subquantizers值: {valid_subquantizers}")
            # 更新IVFPQ参数范围定义
            if valid_subquantizers:
                index_param_ranges["ivfpq"]["num_subquantizers"] = (min(valid_subquantizers), max(valid_subquantizers))
        
        index_type = self.index_config.find_index_type
                
        # 保存优化结果的列表
        optimization_results = []
        
        # 定义目标函数
        def objective(trial):
            # 构建索引参数 - 只探索索引参数
            index_param = {}
            
            # 复制原始的理论参数作为基础
            for key, value in theoretical_params.index_param.items():
                if not key.endswith('_range'):
                    index_param[key] = value
            
            # 针对指定索引类型，让Optuna提供新的索引参数
            for param_name, param_range in index_param_ranges.get(index_type, {}).items():
                # IVFPQ索引类型的num_subquantizers参数需要特殊处理
                if index_type == "ivfpq" and param_name == "num_subquantizers":
                    # 使用discrete_uniform来选择合法的子量化器数量
                    if valid_subquantizers:
                        param_value = trial.suggest_categorical(param_name, valid_subquantizers)
                    else:
                        # 如果没有合法值，使用默认值并记录警告
                        param_value = 1
                        logger.warning(f"使用默认num_subquantizers={param_value}，因为没有找到符合维度{vector_dimension}的合法值")
                else:
                    # 使用理论最优参数作为参考点
                    theoretical_value = index_param.get(param_name, (param_range[0] + param_range[1]) // 2)
                    
                    # 在理论值附近搜索，但不超出范围
                    min_value = max(param_range[0], int(theoretical_value * 0.5))
                    max_value = min(param_range[1], int(theoretical_value * 4.0))
                    
                    param_value = trial.suggest_int(param_name, min_value, max_value)
                
                index_param[param_name] = param_value
                # 添加范围参数
                index_param[f"{param_name}_range"] = param_range
            
            # 使用理论最优的查询参数作为起点
            query_param = {}
            for key, value in theoretical_params.query_param.items():
                query_param[key] = value
            
            # 对HNSW索引类型添加参数约束：ef_construction必须大于等于2*m
            if index_type == "hnsw" and "m" in index_param and "ef_construction" in index_param:
                if index_param["ef_construction"] < 2 * index_param["m"]:
                    logger.info(f"调整HNSW参数：ef_construction从{index_param['ef_construction']}增加到{2 * index_param['m']}，以满足ef_construction >= 2*m的要求")
                    index_param["ef_construction"] = max(index_param["ef_construction"], 2 * index_param["m"])
            
            # 创建参数对象
            param = IndexAndQueryParam(
                index_type=index_type,
                index_param=index_param,
                query_param=query_param
            )
            
            try:
                # 使用analyze_with_param函数分析参数
                # 它会自动为当前索引找到满足召回率要求的最佳查询参数
                result = self.analyze_with_param(param)
                
                # 如果找不到满足召回率的参数，返回惩罚值
                if not result["success"]:
                    logger.warning(f"索引参数 {index_param} 无法满足最低召回率要求")
                    return float('inf')
                
                # 创建一个综合得分 (权重可调整)
                # 索引创建时间权重
                index_time_weight = self.performance.weight["create_index_time"]
                # 查询性能权重
                qps_weight = self.performance.weight["qps"]
                
                # 获取查询时间(毫秒) - 直接使用已有的avg_query_time
                query_time_ms = result["best_performance"]["avg_query_time"] * 1000.0  # 转换为毫秒
                
                # 计算综合得分 - 越小越好
                # 使用更健壮的归一化方法
                # 对于索引创建时间：转换为秒并直接使用
                # 对于查询时间：转换为毫秒并直接使用
                # 这样两个指标的量级更接近，避免某个指标因数值过大或过小而被放大或忽略
                score = (
                    index_time_weight * result["create_index_time"] + 
                    qps_weight * query_time_ms  # 使用毫秒单位的查询时间
                )
                
                # 记录原始指标值以便分析
                normalized_metrics = {
                    "create_index_time_s": result["create_index_time"],
                    "index_size_mb": result["index_size"] / (1024 * 1024),
                    "query_time_ms": query_time_ms,
                    "avg_query_time_s": result["best_performance"]["avg_query_time"],
                    "qps": result["best_performance"]["qps"],
                    "weighted_create_index_time": index_time_weight * result["create_index_time"],
                    "weighted_query_time": qps_weight * query_time_ms
                }
                
                # 保存结果以便后续分析
                trial_result = {
                    "trial_number": trial.number,
                    "index_param": result["index_param"],
                    "best_query_param": result["best_query_param"],
                    "create_index_time": result["create_index_time"],
                    "index_size": result["index_size"],
                    "score": score,
                    "best_performance": result["best_performance"],
                    "normalized_metrics": normalized_metrics  # 添加归一化指标
                }
                optimization_results.append(trial_result)
                
                # 记录本次评估结果以便Optuna分析
                trial.set_user_attr("recall", result["best_performance"]["recall"])
                trial.set_user_attr("create_index_time", result["create_index_time"])
                trial.set_user_attr("index_size", result["index_size"])
                trial.set_user_attr("avg_query_time", result["best_performance"]["avg_query_time"])
                trial.set_user_attr("qps", result["best_performance"]["qps"])
                
                # 记录归一化指标
                for key, value in normalized_metrics.items():
                    trial.set_user_attr(key, value)
                
                # 记录详细的评分组成
                logger.info(f"试验 {trial.number} 评分详情:")
                logger.info(f"  总分: {score:.6f}")
                logger.info(f"  创建索引时间: {result['create_index_time']:.2f}秒 -> 权重贡献: {normalized_metrics['weighted_create_index_time']:.6f}")
                logger.info(f"  查询时间: {normalized_metrics['query_time_ms']:.2f}毫秒 -> 权重贡献: {normalized_metrics['weighted_query_time']:.6f}")
                
                # 保存每次试验的详细结果
                import json
                with open(os.path.join(self.result_dir, f"trial_{trial.number}.json"), "w") as f:
                    json.dump({
                        "trial_number": trial.number,
                        "best_query_param": {k: v for k, v in result["best_query_param"].items() if not k.endswith('_range')},
                        "create_index_time": result["create_index_time"],
                        "index_size": result["index_size"] / (1024 * 1024),  # MB
                        "recall": result["best_performance"]["recall"],
                        "avg_query_time": result["best_performance"]["avg_query_time"],
                        "qps": result["best_performance"]["qps"],
                        "score": score,
                        "normalized_metrics": normalized_metrics  # 添加归一化指标
                    }, f, indent=2)
                
                return score
                
            except Exception as e:
                logger.error(f"参数评估失败: {str(e)}")
                # 记录错误
                with open(os.path.join(self.result_dir, f"error_trial_{trial.number}.txt"), "w") as f:
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"Index params: {index_param}\n")
                return float('inf')
        
        # 创建学习过程 - 使用内存存储而不是SQLite
        study_name = f"optimize_{index_type}_index"
        study = optuna.create_study(
            study_name=study_name,
            direction='minimize'
        )
        
        # 可以添加理论最优解作为初始点
        logger.info("添加理论最优解作为初始点")
        def theoretical_params_func():
            return {
                param_name: theoretical_params.index_param[param_name] 
                for param_name in index_param_ranges.get(index_type, {})
                if param_name in theoretical_params.index_param and not param_name.endswith('_range')
            }
        
        try:
            study.enqueue_trial(theoretical_params_func())
        except Exception as e:
            logger.warning(f"添加理论最优解失败: {e}")
        
        # 运行优化过程
        n_trials = self.config.explore_times  # 可以根据计算资源和时间调整
        logger.info(f"开始索引参数优化，将尝试{n_trials}组索引参数")
        
        try:
            study.optimize(objective, n_trials=n_trials)
        except KeyboardInterrupt:
            logger.warning("用户中断了优化过程")
        except Exception as e:
            logger.error(f"优化过程出错: {e}")
        
        # 获取最佳参数
        if study.best_trial:
            best_params = study.best_params
            best_trial = study.best_trial
            
            logger.info(f"找到的最佳索引参数: {best_params}")
            logger.info(f"最佳参数的评估指标:")
            logger.info(f"- 召回率: {best_trial.user_attrs['recall']:.6f}")
            logger.info(f"- 创建索引时间: {best_trial.user_attrs['create_index_time']:.2f}秒")
            logger.info(f"- 索引大小: {best_trial.user_attrs['index_size']/1024/1024:.2f}MB")
            logger.info(f"- 平均查询时间: {best_trial.user_attrs['avg_query_time']:.6f}秒")
            logger.info(f"- QPS: {best_trial.user_attrs['qps']:.4f}")
            
            # 可视化优化过程
            try:
                # 绘制优化历史
                fig1 = plot_optimization_history(study)
                fig1.write_html(os.path.join(self.result_dir, "optimization_history.html"))
                
                # 绘制参数重要性
                fig2 = plot_param_importances(study)
                fig2.write_html(os.path.join(self.result_dir, "param_importances.html"))
                
                
                logger.info(f"已生成可视化报告在 {self.result_dir} 目录下")
            except Exception as e:
                logger.warning(f"生成可视化报告失败: {e}")
            
            # 保存所有优化结果
            import json
            with open(os.path.join(self.result_dir, "all_results.json"), "w") as f:
                json.dump(
                    [{k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if not kk.endswith('_range')}) 
                      for k, v in result.items()} 
                     for result in optimization_results], 
                    f, indent=2
                )
            
            logger.info(f"优化完成，共探索了{len(optimization_results)}组有效参数")
            
            # 直接返回Optuna找到的最佳参数和性能指标，而不是重新调用analyze_with_param
            best_trial_result = None
            for result in optimization_results:
                if result["trial_number"] == best_trial.number:
                    best_trial_result = result
                    break
            
                
            # 从优化结果中获取完整的参数信息
            best_result = {
                "create_index_time": best_trial.user_attrs["create_index_time"],
                "index_size": best_trial.user_attrs["index_size"],
                "table_size": best_trial.user_attrs["table_size"],
                "index_type": index_type,
                "index_param": best_trial_result['index_param'],
                "best_query_param": best_trial_result["best_query_param"],
                "best_performance": best_trial_result["best_performance"],
                "user_min_recall": self.min_recall,
                "success": bool(best_trial.user_attrs["recall"] >= self.min_recall)
            }
            
            return best_result
        else:
            logger.warning("未能找到有效的参数组合")
            # 如果没有找到任何有效参数，返回理论最优的结果
            return self.analyze_with_param(theoretical_params)


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
                'db_engine_obj': self.db_engine,
                'parallel_workers': self.parallel_workers
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
            "success": False
        }

        # 预热缓存
        if self.index_config.prepare_cache:
            # 获取第一组参数进行预热
            module.set_query_arguments(**query_param)
            # 请所有的请求请求一遍, 预热图索引
            for query_data in self.query_data:
                module.query(query_data.vectors, self.performance.limit)

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
        if recall < self.min_recall:
            logger.info(f"参数组合的召回率 {recall:.6f} 低于最小召回率 {self.min_recall:.6f}，探索更大的参数")
            best_param, best_performance = self._increase_param_until_target(
                module, query_param, param_info)
        
        # 如果当前召回率高于目标值，减小参数值尝试接近目标值
        # 只要高于最低要求就尝试减小参数，但设置一个缓冲区避免频繁波动
        elif recall > self.min_recall + 0.05:  # 设置一个小缓冲区(0.05)避免参数反复波动
            logger.info(f"参数组合的召回率 {recall:.6f} 高于最小召回率 {self.min_recall:.6f}，尝试减小参数提高效率")
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
            "success": bool(best_performance["recall"] >= self.min_recall)
        })
        
        # 过滤掉index_param和best_query_param中的_range参数
        filtered_index_param = {k: v for k, v in analyze_result["index_param"].items() if not k.endswith('_range')}
        filtered_best_query_param = {k: v for k, v in analyze_result["best_query_param"].items() if not k.endswith('_range')}
        
        # 更新过滤后的参数
        analyze_result["index_param"] = filtered_index_param
        analyze_result["best_query_param"] = filtered_best_query_param
        
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
                    "base_step": 20,  # 增加base_step从5到20
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
        
        max_iterations = self.config.explore_times  # 最大迭代次数
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
        
        max_iterations = self.config.explore_times  # 最大迭代次数
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
                # 根据参数名称判断是否是diskann
                is_diskann = primary_name == "diskann_search_list_size"
                
                # 计算召回率与目标值的差距
                recall_margin = best_performance["recall"] - self.min_recall
                
                # 计算主参数的减小步长
                if is_diskann and recall_margin > 0.1:
                    # 对于diskann且召回率远高于目标时，使用更激进的步长
                    # 根据召回率差距动态调整步长
                    margin_factor = min(3.0, max(1.0, recall_margin * 10))  # 差距0.1→1倍，差距0.2→2倍，差距≥0.3→3倍
                    primary_step = int(primary_base_step * margin_factor)
                    # 让步长与当前参数值成比例，更大的参数值可以有更大的减小步长
                    primary_step = max(5, min(current_param[primary_name] // 4, primary_step))
                    logger.info(f"召回率({best_performance['recall']:.6f})远高于目标({self.min_recall:.6f})，使用更大步长: {primary_step}")
                else:
                    # 其他索引类型或差距不大时，使用原来的逻辑
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
            results.append(single_query(module, query_data.vectors, self.performance.limit))

        #计算召回率
        recall = round(get_recall_values([query_data.distances for query_data in self.query_data], [distances for _, distances in results], self.performance.limit), 6)
        
        # 计算平均查询时间
        avg_query_time = round(sum(time for time, _ in results) / len(results), 6)

        # 最小查询时间
        min_query_time = round(min(time for time, _ in results), 6)

        # 最大查询时间
        max_query_time = round(max(time for time, _ in results), 6)

        # 计算QPS
        qps = round(1 / avg_query_time, 4)

        return recall, avg_query_time, min_query_time, max_query_time, qps



