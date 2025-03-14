#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import json
from core.sampling import Sampling
from core.config import config
from core.engine import db_engine
from core.logging import logger
from core.param_builder import IndexParamBuilder

config_path = os.path.join(os.path.dirname(__file__), 'config.yml')

def test_sampling():    
    """测试采样器"""
    logger.info("开始测试采样器")

    config.prepare_with_yaml(config_path)
    
    sampling = Sampling(config, db_engine)

    # 测试采样数据
    sampling.sampling_data()
    logger.info(f"成功采样{sampling.config.table_info.sample_table_count}条数据到表: {sampling.config.table_info.sample_table_name}")

    # 测试采样查询数据
    sampling.sampling_query_data()
    logger.info(f"成功采样{sampling.config.table_info.query_table_count}条查询数据到表: {sampling.config.table_info.query_table_name}")

    # 测试计算最近邻
    sampling.compute_sample_query_distance()
    logger.info("最近邻计算完成")

def test_param_builder():
    """测试参数构建器"""
    sampling = Sampling(config, db_engine)
    config.table_info.sample_table_count = sampling.get_sample_table_count()
    config.table_info.query_table_count = sampling.get_query_table_count()
    print(config.table_info.sample_table_count, config.table_info.query_table_count)
    
    logger.info("开始测试参数构建器")
    param_builder = IndexParamBuilder(config)
    params = param_builder.get_index_params()
    logger.info(f"成功获取参数: {json.dumps(params)}")


if __name__ == "__main__":
    # 运行测试
    # test_sampling()
    test_param_builder()
