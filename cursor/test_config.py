#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试配置类中sample_table_name和query_table_name的值
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vecindex_finder.core.config import Config, TableInfoConfig

def test_default_config():
    """测试默认配置"""
    config = Config()
    print("=== 默认配置 ===")
    print(f"table_name: {config.table_info.table_name}")
    print(f"sample_table_name: {config.table_info.sample_table_name}")
    print(f"query_table_name: {config.table_info.query_table_name}")

def test_custom_table_name():
    """测试自定义表名"""
    table_info = TableInfoConfig(table_name="custom_table")
    config = Config(table_info=table_info)
    print("\n=== 自定义表名 ===")
    print(f"table_name: {config.table_info.table_name}")
    print(f"sample_table_name: {config.table_info.sample_table_name}")
    print(f"query_table_name: {config.table_info.query_table_name}")

def test_explicit_sample_table_name():
    """测试显式设置sample_table_name"""
    table_info = TableInfoConfig(
        table_name="custom_table",
        sample_table_name="explicit_sample_table"
    )
    config = Config(table_info=table_info)
    print("\n=== 显式设置sample_table_name ===")
    print(f"table_name: {config.table_info.table_name}")
    print(f"sample_table_name: {config.table_info.sample_table_name}")
    print(f"query_table_name: {config.table_info.query_table_name}")

def test_model_validation():
    """测试模型验证过程"""
    print("\n=== 模型验证过程 ===")
    # 创建原始数据
    data = {
        "table_name": "test_table",
        "sample_table_name": None,
        "query_table_name": None
    }
    
    # 使用model_validate创建实例
    table_info = TableInfoConfig.model_validate(data)
    print(f"使用model_validate后:")
    print(f"table_name: {table_info.table_name}")
    print(f"sample_table_name: {table_info.sample_table_name}")
    print(f"query_table_name: {table_info.query_table_name}")
    
    # 直接创建实例
    table_info2 = TableInfoConfig(**data)
    print(f"\n使用构造函数后:")
    print(f"table_name: {table_info2.table_name}")
    print(f"sample_table_name: {table_info2.sample_table_name}")
    print(f"query_table_name: {table_info2.query_table_name}")

def test_model_dump():
    """测试模型导出"""
    config = Config()
    print("\n=== 模型导出 ===")
    # 导出为字典
    config_dict = config.model_dump()
    print(f"sample_table_name in dict: {config_dict['table_info']['sample_table_name']}")
    print(f"query_table_name in dict: {config_dict['table_info']['query_table_name']}")

if __name__ == "__main__":
    test_default_config()
    test_custom_table_name()
    test_explicit_sample_table_name()
    test_model_validation()
    test_model_dump() 