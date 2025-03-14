#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import importlib
from ann_results import store_results
from core.config import IndexConfig
from core.engine import DatabaseEngine, db_engine
from core.module import MODULE_MAP, BaseModule
from core.types import QueryData
from core.recall import get_recall_values
from typing import Dict, Any


def instantiate_module(module_name: str, *args) -> BaseModule:
    """
    创建一个 `BaseModule` 对象
    """
    module = importlib.import_module(f"core.module")
    constructor = getattr(module, MODULE_MAP[module_name])
    return constructor(*args)

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
    print(f"创建索引时间: {create_index_time:.2f} 秒")
    index_size = module.get_memory_usage()
    table_size = module.get_table_usage()
    print(f"索引大小: {index_size/1024:0.2f} MB")
    print(f"表大小: {table_size/1024:0.2f} MB")
    return create_index_time, index_size, table_size


class Runner:
    def __init__(self, index_config: IndexConfig, index_param: Dict[str, Any], db_engine_obj: DatabaseEngine, query_data: list[QueryData], limit=100):
        """
        Runner 类用于运行索引创建和查询。

        Args:
            index_config: 索引配置对象
            index_param: 索引参数配置对象
            db_engine_obj: 数据库引擎对象
            query_data: 查询数据表，包含 id、向量、groundtruth的distance,
            limit: 查询数据表中查询的行数
        """
        self.index_config = index_config
        self.index_param = index_param
        self.db_engine = db_engine_obj or db_engine
        self.query_data = query_data
        self.limit = limit


    def run(self):
        
        try:
            module = instantiate_module(self.index_config.find_index_type, **self.index_param)
            create_index_time, index_size, table_size = create_index(module)
        except Exception as e:
            print(f"创建索引 {self.index_param['module']} 时出错: {e}")
            return
    
        # 确保至少有一组查询参数
        query_args = self.index_params.query_args

        descriptor = {
            "create_index_time": create_index_time,
            "index_size": index_size,
            "table_size": table_size,
            "index_type": self.index_config.find_index_type,
            "index_param": self.index_param,
        }


        # 预热缓存
        if self.index_config.prepare_cache:
            # 获取第一组参数进行预热
            query_arguments = query_args[0]
            module.set_query_arguments(*query_arguments)
            # 请所有的请求请求一遍, 预热图索引
            for query_data in self.query_data:
                module.query(query_data.vectors, self.limit)


        for pos, query_arguments in enumerate(query_args):
            print(f"\n运行第 {pos}/{len(query_args)}个参数组: {query_arguments}")
            if query_arguments:
                module.set_query_arguments(*query_arguments)
            
            results = []
            for query_data in self.query_data:
                results.append(single_query(module, query_data.vectors, self.limit))

            #计算召回率
            descriptor["recall"] = get_recall_values(self.query_data.distances, [distances for _, distances in results], self.limit)
            print(f"召回率: {descriptor['recall']:.6f}")
            
            # 计算平均查询时间
            descriptor["avg_query_time"] = sum(time for time, _ in results) / len(results)
            print(f"平均查询时间: {descriptor['avg_query_time']:.6f} 秒")

            # 最小查询时间
            descriptor["min_query_time"] = min(time for time, _ in results)
            print(f"最小查询时间: {descriptor['min_query_time']:.6f} 秒")

            # 最大查询时间
            descriptor["max_query_time"] = max(time for time, _ in results)
            print(f"最大查询时间: {descriptor['max_query_time']:.6f} 秒")

            # 计算QPS
            descriptor["qps"] = 1 / descriptor["avg_query_time"]
            print(f"QPS: {descriptor['qps']:.4f}")


            store_results(
                count=self.limit, 
                definition=module,
                query_arguments=query_arguments,
                attrs=descriptor,
                results=results
            )






