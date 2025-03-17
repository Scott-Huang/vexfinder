#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import multiprocessing
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from core.logging import logger
from core.types import ConnectionConfig, TableInfoConfig, SamplingConfig, QueryConfig, PerformanceConfig, IndexConfig, IndexParamsConfig, InitialExploreParamsConfig


class Config(BaseModel):
    """纯ORM风格配置管理类"""
    connection: ConnectionConfig = Field(default_factory=ConnectionConfig)
    table_info: TableInfoConfig = Field(default_factory=TableInfoConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    index_config: IndexConfig = Field(default_factory=IndexConfig)
    initial_explore_params: InitialExploreParamsConfig = Field(default_factory=InitialExploreParamsConfig)
    manual_index_params: IndexParamsConfig = Field(default_factory=IndexParamsConfig)
    parallel_workers: Optional[int] = None
    explore_times: int = 20
    
    @field_validator('parallel_workers', mode='before')
    @classmethod
    def set_parallel_workers(cls, v):
        """设置并行工作线程数"""
        if v is None:
            return max(1, int(multiprocessing.cpu_count() * 0.75))
        return v
    
    @field_validator('explore_times', mode='before')
    @classmethod
    def set_explore_times(cls, v):
        """设置探索次数"""
        if v is None:
            return 20
        return v

    def prepare_with_yaml(self, config_path) -> 'Config':
        """从YAML文件加载配置"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件 {config_path} 不存在")
        
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                logger.info(f"成功加载配置文件: {config_path}")
                
                # 使用 model_validate 创建一个新的实例
                new_config = self.model_validate(config_dict)
                
                # 将新配置的值复制到当前对象
                for field_name, field_value in new_config.model_dump().items():
                    if hasattr(self, field_name):
                        setattr(self, field_name, field_value)
                
                return self
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    def to_yaml(self, config_path: str) -> None:
        """将配置保存为YAML文件"""
        try:
            with open(config_path, 'w') as f:
                yaml.dump(self.model_dump(exclude_none=True), f, default_flow_style=False)
            logger.info(f"成功保存配置到文件: {config_path}")
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
            raise
    
    model_config = {
        "extra": "ignore",
        "validate_assignment": True,
        "arbitrary_types_allowed": True
    }


# 创建全局配置实例
config = Config()
