#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import logging


from sampling import Sampling
from core.config import Config
from core.engine import DatabaseEngine
from core.logging import logger


def test_sampling():
    """测试采样器"""
    logger.info("开始测试采样器")
    
    config = Config()
    db_engine = DatabaseEngine(config)
    sampling = Sampling(config, db_engine)

    # 测试采样数据
    sample_table = sampling.sampling_data()
    logger.info(f"成功采样数据到表: {sample_table}")

    # 测试采样查询数据
    query_table = sampling.sampling_query_data()
    logger.info(f"成功采样{sampling.query_count}条查询数据到表: {query_table}")

    # 测试计算最近邻
    sampling.compute_sample_query_distance()
    logger.info("最近邻计算完成")

if __name__ == "__main__":
    # 运行测试
    test_sampling()
