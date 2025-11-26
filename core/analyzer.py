#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import importlib
import os
import datetime

from core.config import Config
from core.engine import DatabaseEngine, db_engine
from core.module import MODULE_MAP, BaseModule
from core.my_types import QueryData, IndexAndQueryParam
from core.recall import get_recall_values
from core.param_builder import get_theoretical_param
from core.logging import logger
from typing import Dict, Any
from tqdm import tqdm
import copy
import json
import optuna

def instantiate_module(module_name: str, *args, **kwargs) -> BaseModule:
    module = importlib.import_module(f"core.module")
    constructor = getattr(module, MODULE_MAP[module_name])
    return constructor(*args, **kwargs)

def single_query(algo, v, count):
    v_str = algo.preprocess_query(v)
    start = time.time()
    distances = algo.query(v_str, count)
    query_time = time.time() - start
    return (query_time, distances)

def create_index(module):
    start = time.time()
    module.create_index()
    create_index_time = time.time() - start
    logger.info(f"创建索引时间: {create_index_time:.2f} 秒")
    index_size = module.get_index_usage()
    table_size = module.get_table_usage()
    logger.info(f"索引大小: {index_size:0.2f} MB")
    logger.info(f"表大小: {table_size:0.2f} MB")
    return create_index_time, index_size, table_size

class Analyzer:
    def __init__(self, db_engine_obj: DatabaseEngine, query_data: list[QueryData],  config_obj: Config):
        self.config = config_obj
        self.index_config = config_obj.index_config
        self.db_engine = db_engine_obj or db_engine
        self.query_data = query_data
        self.performance = config_obj.performance
        self.min_recall = config_obj.performance.min_recall
        self.table_info = config_obj.table_info
        self.parallel_workers = config_obj.parallel_workers
        self.initial_explore_params = config_obj.initial_explore_params

        if not os.path.isdir(os.path.join(self.config.output_dir)):
            os.makedirs(os.path.join(self.config.output_dir))

        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.result_dir = os.path.join(self.config.output_dir, f"{self.index_config.find_index_type}_{timestamp}")
        if not os.path.isdir(self.result_dir):
            os.makedirs(self.result_dir)

    def analyze(self):
        if not self.index_config.auto:
            return self.run_with_manual_index_params()

        if self.initial_explore_params.manual_param:
            if not self.initial_explore_params.index_param or not self.initial_explore_params.query_param:
                raise ValueError("当前initial_explore_params配置的 manual_param 为true，索引参数和查询参数不能为空")
            theoretical_params = IndexAndQueryParam(
                index_type=self.index_config.find_index_type,
                index_param=self.initial_explore_params.index_param,
                query_param=self.initial_explore_params.query_param
            )
            logger.info(f"使用用户设置参数作为优化起点: {theoretical_params}")
        else:
            theoretical_params = get_theoretical_param(self.table_info.sample_table_count, 
                                                       self.index_config.find_index_type)
            logger.info(f"获取理论最优参数作为优化起点: {theoretical_params}")

        is_index_param_empty = not theoretical_params.index_param or all(
            k.endswith('_range') for k in theoretical_params.index_param.keys()
        )

        if is_index_param_empty:
            return self.run_empty_index_param(theoretical_params)
        else:
            return self.run_with_optuna(theoretical_params)

    def run_with_manual_index_params(self):
        if not self.config.manual_index_params:
                raise ValueError("当前index_config配置的auto为false，需要设置manual_index_params")

        manual_index_params = self.config.manual_index_params
        all_results = []
        best_result = None
        best_performance = None

        for param_group in manual_index_params:
            if "index_param" not in param_group:
                raise ValueError(f"参数组缺少index_param: {param_group}")

            index_param = param_group["index_param"]
            query_param = None
            if "default_query_param" in param_group:
                query_param = param_group["default_query_param"]
            else:
                logger.warning(f"参数组缺少default_query_param: {param_group}")
                continue

            try:
                param = IndexAndQueryParam(
                    index_type=self.index_config.find_index_type,
                    index_param=index_param,
                    query_param=query_param
                )

                logger.info(f"测试参数组合: index_param={index_param}, query_param={query_param}")
                result = self.analyze_with_param(param)
                result["table_info"] = self.table_info.model_dump(mode="json")
                all_results.append(result)

                if not result["success"]:
                    logger.info(f"参数组合未达到最低召回率要求: {self.min_recall}")
                    continue
                if best_result is None or self._is_better_performance(result, best_result):
                    best_result = result
                    best_performance = result["best_performance"]
                    logger.info(f"找到更优的参数组合: 召回率={best_performance['recall']:.4f}, 查询时间={best_performance['avg_query_time']:.4f}秒, QPS={best_performance['qps']:.4f}")
            
            except Exception as e:
                logger.error(f"测试参数组合失败: {str(e)}")

        with open(os.path.join(self.result_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"已保存所有测试结果到 {os.path.join(self.result_dir, 'all_results.json')}")

        if best_result:
            with open(os.path.join(self.result_dir, "best_result.json"), "w") as f:
                json.dump(best_result, f, indent=2)
            logger.info(f"已保存最佳结果到 {os.path.join(self.result_dir, 'best_result.json')}")
            return best_result
        else:
            logger.warning("给出的参数组合中未找到满足召回率要求的参数组合")
            return None

    def _is_better_performance(self, new_result, current_best):
        if not new_result["success"] or not current_best["success"]:
            return new_result["success"]

        index_size_weight = self.performance.weight["index_size"]
        index_time_weight = self.performance.weight["create_index_time"]
        qps_weight = self.performance.weight["qps"]

        new_query_time_ms = new_result["best_performance"]["avg_query_time"] * 1000.0
        new_score = (
            index_size_weight * new_result["index_size"] / 1024 +
            index_time_weight * new_result["create_index_time"] / 60 + 
            qps_weight * new_query_time_ms
        )

        current_query_time_ms = current_best["best_performance"]["avg_query_time"] * 1000.0
        current_score = (
            index_size_weight * current_best["index_size"] / 1024 +
            index_time_weight * current_best["create_index_time"] / 60 + 
            qps_weight * current_query_time_ms
        )
        return new_score < current_score

    def run_empty_index_param(self, param: IndexAndQueryParam):
        logger.info("索引参数为空，仅需搜索查询参数，跳过索引参数优化过程")
        result = self.analyze_with_param(param)
        result["table_info"] = self.table_info.model_dump(mode="json")
        with open(os.path.join(self.result_dir, "best_result.json"), "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"已保存最佳结果到 {os.path.join(self.result_dir, 'best_result.json')}")
        return result

    def run_with_optuna(self, theoretical_params: Dict[str, Any]):
        index_param_ranges = {
            "graph_index": {
                "ef_construction": (40, 800),
                "m": (10, 96),
                # "quantizer": ["\"none\"", "pq", "rabitq"]
                "quantizer": ["\"none\""]
            },
            "diskann": {
                "m": (60, 200),
                "ef_construction": (80, 800),
                "occlusion_factor": (1.01, 1.2)
            },
            "ivfflat": {
                "ivf_nlist": (10, 2000),
            },
            "ivfpq": {
                "ivf_nlist": (10, 2000),
                "num_subquantizers": (1, 64),
            }
        }

        vector_dimension = self.table_info.dimension
        logger.info(f"向量维度: {vector_dimension}")

        if self.index_config.find_index_type == "ivfpq":
            valid_subquantizers = []
            for i in range(1, min(64, vector_dimension) + 1):
                if vector_dimension % i == 0:
                    valid_subquantizers.append(i)
            
            logger.info(f"合法的num_subquantizers值: {valid_subquantizers}")
            index_param_ranges["ivfpq"]["num_subquantizers"] = valid_subquantizers
        
        index_type = self.index_config.find_index_type
        optimization_results = []
        def objective(trial):
            index_param = {}
            for key, value in theoretical_params.index_param.items():
                if not key.endswith('_range'):
                    index_param[key] = value

            for param_name, param_range in index_param_ranges.get(index_type, {}).items():
                if type(param_range) == list:
                    param_value = trial.suggest_categorical(param_name, param_range)
                else:
                    theoretical_value = index_param.get(param_name, (param_range[0] + param_range[1]) // 2)
                    min_value = max(param_range[0], int(theoretical_value * 0.5))
                    max_value = min(param_range[1], int(theoretical_value * 4.0))
                    param_value = trial.suggest_int(param_name, min_value, max_value)

                index_param[param_name] = param_value
                index_param[f"{param_name}_range"] = param_range
            if "m" in index_param and "ef_construction" in index_param:
                if index_type == "graph_index":
                    index_param["ef_construction"] = max(index_param["ef_construction"], 2 * index_param["m"])
                elif index_type == "diskann":
                    index_param["ef_construction"] = max(index_param["ef_construction"], 10 + index_param["m"])

            query_param = {}
            for key, value in theoretical_params.query_param.items():
                query_param[key] = value
            param = IndexAndQueryParam(
                index_type=index_type,
                index_param=index_param,
                query_param=query_param
            )

            try:
                result = self.analyze_with_param(param)
                if not result["success"]:
                    logger.warning(f"索引参数 {index_param} 无法满足最低召回率要求")
                    return float('inf')

                index_size_weight = self.performance.weight["index_size"]
                index_time_weight = self.performance.weight["create_index_time"]
                qps_weight = self.performance.weight["qps"]
                query_time_ms = result["best_performance"]["avg_query_time"] * 1000.0

                score = \
                    index_size_weight * result["index_size"] / 1024 + \
                    index_time_weight * result["create_index_time"] / 60 + \
                    qps_weight * query_time_ms

                normalized_metrics = {
                    "create_index_time_s": result["create_index_time"],
                    "index_size_mb": result["index_size"],
                    "table_size_mb": result["table_size"],
                    "query_time_ms": query_time_ms,
                    "avg_query_time_s": result["best_performance"]["avg_query_time"],
                    "qps": result["best_performance"]["qps"],
                    "weighted_index_size": index_size_weight * result["index_size"] / 1024,
                    "weighted_create_index_time": index_time_weight * result["create_index_time"] / 60,
                    "weighted_query_time": qps_weight * query_time_ms
                }

                trial_result = {
                    "trial_number": trial.number,
                    "index_param": result["index_param"],
                    "best_query_param": result["best_query_param"],
                    "create_index_time": result["create_index_time"],
                    "index_size": result["index_size"],
                    "table_size": result["table_size"],
                    "score": score,
                    "best_performance": result["best_performance"],
                    "normalized_metrics": normalized_metrics  # 添加归一化指标
                }
                optimization_results.append(trial_result)

                trial.set_user_attr("recall", result["best_performance"]["recall"])
                trial.set_user_attr("create_index_time", result["create_index_time"])
                trial.set_user_attr("index_size", result["index_size"])
                trial.set_user_attr("table_size", result["table_size"])
                trial.set_user_attr("avg_query_time", result["best_performance"]["avg_query_time"])
                trial.set_user_attr("qps", result["best_performance"]["qps"])

                for key, value in normalized_metrics.items():
                    trial.set_user_attr(key, value)

                logger.info(f"试验 {trial.number} 评分详情:")
                logger.info(f"  总分: {score:.4f}")
                logger.info(f"  索引大小: {result["index_size"]}MB -> 权重贡献: {normalized_metrics['weighted_index_size']:.4f}")
                logger.info(f"  创建索引时间: {result['create_index_time']/60:.2f}分钟 -> 权重贡献: {normalized_metrics['weighted_create_index_time']:.4f}")
                logger.info(f"  查询时间: {normalized_metrics['query_time_ms']:.2f}毫秒 -> 权重贡献: {normalized_metrics['weighted_query_time']:.4f}")

                with open(os.path.join(self.result_dir, f"trial_{trial.number}.json"), "w") as f:
                    json.dump({
                        "trial_number": trial.number,
                        "table_info": self.table_info.model_dump(mode="json"),
                        "best_query_param": {k: v for k, v in result["best_query_param"].items() if not k.endswith('_range')},
                        "create_index_time": result["create_index_time"],
                        "index_size": result["index_size"],  # MB
                        "table_size": result["table_size"],  # MB
                        "recall": result["best_performance"]["recall"],
                        "avg_query_time": result["best_performance"]["avg_query_time"],
                        "qps": result["best_performance"]["qps"],
                        "score": score,
                        "index_param": result["index_param"],
                        "best_performance": result["best_performance"],
                        "find_index_type": index_type,
                    }, f, indent=2)
                return score
            except Exception as e:
                logger.error(f"参数评估失败: {str(e)}")
                with open(os.path.join(self.result_dir, f"error_trial_{trial.number}.txt"), "w") as f:
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"Index params: {index_param}\n")
                return float('inf')

        study_name = f"optimize_{index_type}_index"
        study = optuna.create_study(
            study_name=study_name,
            direction='minimize'
        )

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

        n_trials = self.config.explore_times
        logger.info(f"开始索引参数优化，将尝试{n_trials}组索引参数")

        try:
            study.optimize(objective, n_trials=n_trials)
        except KeyboardInterrupt:
            logger.warning("用户中断了优化过程")
        except Exception as e:
            logger.error(f"优化过程出错: {e}")

        if study.best_trial:
            best_params = study.best_params
            best_trial = study.best_trial
            logger.info(f"找到的最佳索引参数: {best_params}")
            logger.info(f"最佳参数的评估指标:")
            logger.info(f"- 召回率: {best_trial.user_attrs['recall']:.4f}")
            logger.info(f"- 创建索引时间: {best_trial.user_attrs['create_index_time']/60:.2f}分钟")
            logger.info(f"- 索引大小: {best_trial.user_attrs['index_size']:.2f}MB")
            logger.info(f"- 表大小: {best_trial.user_attrs['table_size']:.2f}MB")
            logger.info(f"- 平均查询时间: {best_trial.user_attrs['avg_query_time']:.6f}秒")
            logger.info(f"- QPS: {best_trial.user_attrs['qps']:.4f}")

            try:
                fig1 = optuna.visualization.plot_optimization_history(study)
                fig1.write_html(os.path.join(self.result_dir, "optimization_history.html"))
                fig2 = optuna.visualization.plot_param_importances(study)
                fig2.write_html(os.path.join(self.result_dir, "param_importances.html"))
                logger.info(f"查找过程已生成可视化报告在 {self.result_dir} 目录下")
            except Exception as e:
                logger.warning(f"查找过程生成可视化报告失败: {e}")

            with open(os.path.join(self.result_dir, "all_results.json"), "w") as f:
                json.dump(
                    [{k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if not kk.endswith('_range')}) 
                      for k, v in result.items()} 
                     for result in optimization_results], 
                    f, indent=2
                )

            logger.info(f"优化完成，共探索了{len(optimization_results)}组有效参数")

            best_trial_result = None
            for result in optimization_results:
                if result["trial_number"] == best_trial.number:
                    best_trial_result = result
                    break

            best_result = {
                "table_info": self.table_info.model_dump(mode="json"),
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

            with open(os.path.join(self.result_dir, "best_result.json"), "w") as f:
                json.dump(best_result, f, indent=2)
            logger.info(f"已保存最佳结果到 {os.path.join(self.result_dir, 'best_result.json')}")
            return best_result
        else:
            logger.warning("未能找到有效的参数组合")
            raise ValueError("未能找到有效的参数组合，请检查配置文件和数据")

    def analyze_with_param(self, param: IndexAndQueryParam):
        index_param = param.index_param
        query_param = param.query_param
        if param.index_type != self.index_config.find_index_type:
            raise ValueError(f"索引类型不匹配: {param.index_type} != {self.index_config.find_index_type}")

        try:
            if not self.table_info:
                raise ValueError("表信息配置为空")
            full_params = {
                'table_name': self.table_info.sample_table_name,
                'vector_column_name': self.table_info.vector_column_name,
                'metric': self.table_info.metric,
                'db_engine_obj': self.db_engine,
                'parallel_workers': self.parallel_workers
            }

            filtered_index_param = {k: v for k, v in index_param.items() if not k.endswith('_range')}

            full_params.update(filtered_index_param)
            module = instantiate_module(self.index_config.find_index_type, **full_params)
            create_index_time, index_size, table_size = create_index(module)
        except Exception as e:
            raise ValueError(f"创建索引 {self.index_config.find_index_type} 时出错: {e}")

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

        if self.index_config.prepare_cache:
            module.set_query_arguments(**query_param)
            logger.info("预热缓存准备查询")
            for _ in tqdm(range(5)):
                for query_data in tqdm(self.query_data, leave=False):
                    module.query(module.preprocess_query(query_data.vectors), self.performance.limit)

        recall, avg_query_time, min_query_time, max_query_time, qps = self.test_query_param(module, query_param)
        best_param = copy.deepcopy(query_param)
        best_performance = {
            "recall": recall,
            "avg_query_time": avg_query_time,
            "min_query_time": min_query_time,
            "max_query_time": max_query_time,
            "qps": qps
        }

        if recall < self.min_recall:
            logger.info(f"参数组合的召回率 {recall:.4f} 低于最小召回率 {self.min_recall:.4f}，探索更大的参数")
            best_param, best_performance = self._increase_param_until_target(module, query_param, param_info)
        elif recall > self.min_recall + 0.05:
            logger.info(f"参数组合的召回率 {recall:.4f} 高于最小召回率 {self.min_recall:.4f}，尝试减小参数提高效率")
            best_param, best_performance = self._decrease_param_until_target(
                module, query_param, param_info)

        logger.info(f"最佳参数: {best_param}")
        logger.info(f"最佳性能: 召回率={best_performance['recall']:.4f}, " 
              f"平均查询时间={best_performance['avg_query_time']:.6f}秒, "
              f"QPS={best_performance['qps']:.4f}")

        analyze_result.update({
            "best_query_param": best_param,
            "best_performance": best_performance,
            "success": bool(best_performance["recall"] >= self.min_recall)
        })

        filtered_index_param = {k: v for k, v in analyze_result["index_param"].items() if not k.endswith('_range')}
        filtered_best_query_param = {k: v for k, v in analyze_result["best_query_param"].items() if not k.endswith('_range')}
        analyze_result["index_param"] = filtered_index_param
        analyze_result["best_query_param"] = filtered_best_query_param
        return analyze_result

    def _get_param_strategy(self, index_type: str, query_param: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        if index_type == "graph_index":
            result = {
                "primary_param": {
                    "name": "hnsw_ef_search",
                    "base_step": 10,
                    "min_value": query_param.get("hnsw_ef_search_range", [5, 2000])[0],
                    "max_value": query_param.get("hnsw_ef_search_range", [5, 2000])[1]
                }
            }
        elif index_type == "ivfflat":
            result = {
                "primary_param": {
                    "name": "ivf_probes",
                    "base_step": 5,
                    "min_value": query_param.get("ivf_probes_range", [2, 10000])[0],
                    "max_value": query_param.get("ivf_probes_range", [2, 10000])[1]
                }
            }
        elif index_type == "ivfpq":
            result = {
                "primary_param": {
                    "name": "ivf_probes",
                    "base_step": 5,
                    "min_value": query_param.get("ivf_probes_range", [2, 10000])[0],
                    "max_value": query_param.get("ivf_probes_range", [2, 10000])[1]
                },
                "secondary_param": {
                    "name": "ivfpq_refine_k_factor",
                    "base_step": 1,
                    "min_value": query_param.get("ivfpq_refine_k_factor_range", [2, 64])[0],
                    "max_value": query_param.get("ivfpq_refine_k_factor_range", [2, 64])[1]
                }
            }
        elif index_type == "diskann":
            result = {
                "primary_param": {
                    "name": "diskann_search_list_size",
                    "base_step": 20,
                    "min_value": query_param.get("diskann_search_list_size_range", [5, 2000])[0],
                    "max_value": query_param.get("diskann_search_list_size_range", [5, 2000])[1]
                }
            }
        return result

    def _calculate_step(self, current_param_value: int, current_recall: float, min_recall: float, base_step: int) -> int:
        if current_recall < min_recall:
            recall_diff = min_recall - current_recall
            step_factor = max(0.1, min(1.0, recall_diff / 0.1))
            adjusted_step = max(base_step, int(current_param_value * step_factor))
            return adjusted_step
        else:
            return max(1, base_step // 2)

    def _increase_param_until_target(self, module, query_param, param_info):
        current_param = copy.deepcopy(query_param)
        best_param = copy.deepcopy(query_param)

        primary_param = param_info["primary_param"]
        primary_name = primary_param["name"]
        primary_base_step = primary_param["base_step"]
        primary_min = primary_param["min_value"]
        primary_max = primary_param["max_value"]

        has_secondary = "secondary_param" in param_info
        if has_secondary:
            secondary_param = param_info["secondary_param"]
            secondary_name = secondary_param["name"]
            secondary_base_step = secondary_param["base_step"]
            secondary_min = secondary_param["min_value"]
            secondary_max = secondary_param["max_value"]

        initial_recall, initial_avg_query_time, initial_min_query_time, initial_max_query_time, initial_qps = \
            self.test_query_param(module, current_param)

        best_performance = {
            "recall": initial_recall,
            "avg_query_time": initial_avg_query_time,
            "min_query_time": initial_min_query_time,
            "max_query_time": initial_max_query_time,
            "qps": initial_qps
        }

        if has_secondary:
            logger.info(f"初始参数 {primary_name}={current_param[primary_name]}, {secondary_name}={current_param[secondary_name]}, 召回率={initial_recall:.4f}, 查询时间={initial_avg_query_time:.4f}秒, QPS={initial_qps:.4f}")
        else:
            logger.info(f"初始参数 {primary_name}={current_param[primary_name]}, 召回率={initial_recall:.4f}, 查询时间={initial_avg_query_time:.4f}秒, QPS={initial_qps:.4f}")

        max_iterations = self.config.explore_times
        iterations = 0

        primary_only_phase = True
        secondary_adjusted = False

        while iterations < max_iterations:
            iterations += 1
            primary_step = self._calculate_step(
                current_param[primary_name], 
                best_performance["recall"], 
                self.min_recall,
                primary_base_step
            )

            new_primary_value = min(primary_max, current_param[primary_name] + primary_step)
            if (new_primary_value == current_param[primary_name] == primary_max and 
                best_performance["recall"] < self.min_recall and 
                has_secondary and primary_only_phase):
                logger.info(f"主参数 {primary_name} 已达最大值 {primary_max}，开始调整次要参数 {secondary_name}")
                primary_only_phase = False
            current_param[primary_name] = new_primary_value

            if has_secondary and not primary_only_phase:
                secondary_step = self._calculate_step(
                    current_param[secondary_name], 
                    best_performance["recall"], 
                    self.min_recall,
                    secondary_base_step
                )

                new_secondary_value = min(secondary_max, current_param[secondary_name] + secondary_step)
                current_param[secondary_name] = new_secondary_value
                secondary_adjusted = True
                logger.info(f"使用动态步长，将 {primary_name} 增加到 {new_primary_value}，{secondary_name} 增加到 {new_secondary_value}")
            else:
                logger.info(f"使用动态步长 {primary_step}，将 {primary_name} 增加到 {new_primary_value}")

            if (current_param[primary_name] == best_param[primary_name] and 
                (not has_secondary or current_param[secondary_name] == best_param[secondary_name])):
                logger.info(f"参数已达上限，无法继续增加")
                break

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
                logger.info(f"参数 {primary_name}={current_param[primary_name]}, {secondary_name}={current_param[secondary_name]}, 召回率={recall:.4f}, 查询时间={avg_query_time:.4f}秒, QPS={qps:.4f}")
            else:
                logger.info(f"参数 {primary_name}={current_param[primary_name]}, 召回率={recall:.4f}, 查询时间={avg_query_time:.4f}秒, QPS={qps:.4f}")

            if recall >= self.min_recall:
                best_param = copy.deepcopy(current_param)
                best_performance = current_performance
                logger.info(f"找到满足目标召回率的参数组合, 召回率={recall:.4f}")
                break

            if recall - best_performance["recall"] < 0.01:
                logger.info(f"召回率增长缓慢（小于1%），可能已接近上限")
                if has_secondary and primary_only_phase:
                    logger.info(f"尝试调整次要参数 {secondary_name} 以进一步提高召回率")
                    primary_only_phase = False
                    continue

            if recall > best_performance["recall"]:
                best_param = copy.deepcopy(current_param)
                best_performance = current_performance

        if best_performance["recall"] < self.min_recall:
            logger.info(f"警告：在{max_iterations}次迭代后仍未找到满足目标召回率 {self.min_recall} 的参数")
            logger.info(f"最佳找到的召回率为 {best_performance['recall']:.4f}")

            if has_secondary and not secondary_adjusted:
                logger.info(f"尝试调整次要参数 {secondary_name} 以进一步提高召回率")
                current_param = copy.deepcopy(best_param)
                for i in range(5):
                    secondary_step = max(1, secondary_base_step * (i + 1))
                    new_secondary_value = min(secondary_max, current_param[secondary_name] + secondary_step)
                    if new_secondary_value == current_param[secondary_name]:
                        break
                    current_param[secondary_name] = new_secondary_value
                    logger.info(f"增加次要参数 {secondary_name} 到 {new_secondary_value}")

                    recall, avg_query_time, min_query_time, max_query_time, qps = \
                        self.test_query_param(module, current_param)

                    logger.info(f"参数 {primary_name}={current_param[primary_name]}, {secondary_name}={current_param[secondary_name]}, 召回率={recall:.4f}, 查询时间={avg_query_time:.4f}秒, QPS={qps:.4f}")

                    if recall > best_performance["recall"]:
                        best_param = copy.deepcopy(current_param)
                        best_performance = {
                            "recall": recall,
                            "avg_query_time": avg_query_time,
                            "min_query_time": min_query_time,
                            "max_query_time": max_query_time,
                            "qps": qps
                        }
                        logger.info(f"次要参数调整改进了召回率，新召回率={recall:.4f}")
                        if recall >= self.min_recall:
                            logger.info(f"找到满足目标召回率的参数组合")
                            break

        return best_param, best_performance

    def _decrease_param_until_target(self, module, query_param, param_info):
        current_param = copy.deepcopy(query_param)
        best_param = copy.deepcopy(query_param)

        primary_param = param_info["primary_param"]
        primary_name = primary_param["name"]
        primary_base_step = primary_param["base_step"]
        primary_min = primary_param["min_value"]
        primary_max = primary_param["max_value"]

        has_secondary = "secondary_param" in param_info
        if has_secondary:
            secondary_param = param_info["secondary_param"]
            secondary_name = secondary_param["name"]
            secondary_base_step = secondary_param["base_step"]
            secondary_min = secondary_param["min_value"]
            secondary_max = secondary_param["max_value"]

        recall, avg_query_time, min_query_time, max_query_time, qps = \
            self.test_query_param(module, current_param)

        best_performance = {
            "recall": recall,
            "avg_query_time": avg_query_time,
            "min_query_time": min_query_time,
            "max_query_time": max_query_time,
            "qps": qps
        }

        if has_secondary:
            logger.info(f"初始参数 {primary_name}={current_param[primary_name]}, {secondary_name}={current_param[secondary_name]}, 召回率={recall:.4f}, 查询时间={avg_query_time:.4f}秒, QPS={qps:.4f}")
        else:
            logger.info(f"初始参数 {primary_name}={current_param[primary_name]}, 召回率={recall:.4f}, 查询时间={avg_query_time:.4f}秒, QPS={qps:.4f}")

        max_iterations = self.config.explore_times
        iterations = 0
        secondary_phase = has_secondary
        while iterations < max_iterations:
            iterations += 1
            if secondary_phase:
                secondary_step = self._calculate_step(
                    current_param[secondary_name], best_performance["recall"], 
                    self.min_recall, secondary_base_step
                ) // 2

                secondary_step = max(1, secondary_step)
                new_secondary_value = max(secondary_min, current_param[secondary_name] - secondary_step)
                if new_secondary_value == current_param[secondary_name] == secondary_min:
                    logger.info(f"次要参数 {secondary_name} 已达最小值 {secondary_min}，开始调整主参数 {primary_name}")
                    secondary_phase = False
                    continue

                current_param[secondary_name] = new_secondary_value
                logger.info(f"减小次要参数 {secondary_name} 到 {new_secondary_value}")
            else:
                is_diskann = primary_name == "diskann_search_list_size"
                recall_margin = best_performance["recall"] - self.min_recall
                if is_diskann and recall_margin > 0.1:
                    margin_factor = min(3.0, max(1.0, recall_margin * 10))
                    primary_step = int(primary_base_step * margin_factor)
                    primary_step = max(5, min(current_param[primary_name] // 4, primary_step))
                    logger.info(f"召回率({best_performance['recall']:.4f})远高于目标({self.min_recall:.4f})，使用更大步长: {primary_step}")
                else:
                    primary_step = self._calculate_step(
                        current_param[primary_name], best_performance["recall"], 
                        self.min_recall, primary_base_step
                    ) // 2

                primary_step = max(1, primary_step)
                new_primary_value = max(primary_min, current_param[primary_name] - primary_step)

                if new_primary_value == current_param[primary_name] == primary_min:
                    logger.info(f"主参数 {primary_name} 已达最小值 {primary_min}，无法继续减小")
                    break

                current_param[primary_name] = new_primary_value
                logger.info(f"减小主参数 {primary_name} 到 {new_primary_value}")

            if (not secondary_phase and current_param[primary_name] == best_param[primary_name]) or \
               (secondary_phase and current_param[secondary_name] == best_param[secondary_name]):
                continue

            recall, avg_query_time, min_query_time, max_query_time, qps = \
                self.test_query_param(module, current_param)

            current_performance = {
                "recall": recall,
                "avg_query_time": avg_query_time,
                "min_query_time": min_query_time,
                "max_query_time": max_query_time,
                "qps": qps
            }

            if has_secondary:
                logger.info(f"参数 {primary_name}={current_param[primary_name]}, {secondary_name}={current_param[secondary_name]}, 召回率={recall:.4f}, 查询时间={avg_query_time:.4f}秒, QPS={qps:.4f}")
            else:
                logger.info(f"参数 {primary_name}={current_param[primary_name]}, 召回率={recall:.4f}, 查询时间={avg_query_time:.4f}秒, QPS={qps:.4f}")

            if recall >= self.min_recall:
                best_param = copy.deepcopy(current_param)
                best_performance = current_performance
                logger.info(f"找到更优的参数组合, 召回率仍满足要求={recall:.4f}")
            else:
                logger.info(f"召回率 {recall:.4f} 低于目标值 {self.min_recall}，回到上一个参数组合")
                if iterations == 1:
                    if secondary_phase:
                        current_param[secondary_name] = best_param[secondary_name]
                        secondary_phase = False
                        continue
                break

        logger.info(f"减小参数搜索结束，选择的最佳参数组合的召回率={best_performance['recall']:.4f}")
        return best_param, best_performance

    def test_query_param(self, module, query_param: Dict[str, Any]):
        filtered_query_param = {k: v for k, v in query_param.items() if not k.endswith('_range')}
        module.set_query_arguments(**filtered_query_param)
        results = []
        for query_data in self.query_data:
            results.append(single_query(module, query_data.vectors, self.performance.limit))
        recall = get_recall_values([query_data.distances for query_data in self.query_data],
                                   [distances for _, distances in results],
                                   self.performance.limit)
        recall = round(recall, 6)

        avg_query_time = round(sum(time for time, _ in results) / len(results), 6)
        min_query_time = round(min(time for time, _ in results), 6)
        max_query_time = round(max(time for time, _ in results), 6)
        qps = round(1 / avg_query_time, 4)
        return recall, avg_query_time, min_query_time, max_query_time, qps
