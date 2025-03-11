#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试数据库连接管理功能
"""

import logging
import sys
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vecindex_finder.core.engine import db_engine
from vecindex_finder.core.config import OpenGaussConfig
from vecindex_finder.db_adapters.opengauss.adapter import OpenGaussAdapter

def test_config():
    """测试配置管理"""
    logger.info("测试配置管理")
    
    try:
        # 创建配置管理器
        config_manager = OpenGaussConfig()
        
        # 获取配置
        connection_config = config_manager.get_connection_config()
        table_info = config_manager.get_table_info()
        index_config = config_manager.get_index_config()
        performance_config = config_manager.get_performance_config()
        
        # 输出配置信息
        logger.info(f"连接配置: {connection_config}")
        logger.info(f"表信息: {table_info}")
        logger.info(f"索引配置: {index_config}")
        logger.info(f"性能配置: {performance_config}")
        
        logger.info("配置管理测试成功")
        return True
    except Exception as e:
        logger.error(f"配置管理测试失败: {e}")
        return False

def test_db_engine():
    """测试数据库引擎"""
    logger.info("测试数据库引擎")
    
    try:
        # 获取配置管理器
        config_manager = db_engine.get_config_manager()
        logger.info(f"配置管理器: {config_manager}")
        
        # 获取连接
        connection = db_engine.get_connection()
        cursor = connection.cursor()
        
        # 执行简单查询
        cursor.execute("SELECT version()")
        version = cursor.fetchone()[0]
        logger.info(f"数据库版本: {version}")
        
        # 关闭连接
        db_engine.close_connection(connection, cursor)
        logger.info("数据库引擎测试成功")
        return True
    except Exception as e:
        logger.error(f"数据库引擎测试失败: {e}")
        return False

def test_adapter():
    """测试适配器"""
    logger.info("测试适配器")
    
    try:
        # 创建适配器
        adapter = OpenGaussAdapter()
        
        # 输出配置信息
        logger.info(f"表名: {adapter.table_name}")
        logger.info(f"向量列名: {adapter.vector_column_name}")
        logger.info(f"维度: {adapter.dimension}")
        logger.info(f"度量类型: {adapter.metric}")
        logger.info(f"并行工作线程数: {adapter.parallel_workers}")
        
        # 连接数据库
        adapter.connect()
        
        # 执行简单查询
        connection = adapter.connection
        cursor = adapter.cursor
        cursor.execute("SELECT version()")
        version = cursor.fetchone()[0]
        logger.info(f"数据库版本: {version}")
        
        # 关闭连接
        adapter.close()
        logger.info("适配器测试成功")
        return True
    except Exception as e:
        logger.error(f"适配器测试失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("开始测试数据库连接管理功能")
    
    # 测试配置管理
    config_result = test_config()
    
    # 测试数据库引擎
    engine_result = test_db_engine()
    
    # 测试适配器
    adapter_result = test_adapter()
    
    # 输出测试结果
    if config_result and engine_result and adapter_result:
        logger.info("所有测试通过")
    else:
        logger.error("测试失败")
        if not config_result:
            logger.error("配置管理测试失败")
        if not engine_result:
            logger.error("数据库引擎测试失败")
        if not adapter_result:
            logger.error("适配器测试失败")

if __name__ == "__main__":
    main() 