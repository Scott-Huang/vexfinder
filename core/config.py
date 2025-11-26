#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import multiprocessing
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from core.logging import logger
from core.types import ConnectionConfig, TableInfoConfig, SamplingConfig, QueryConfig, PerformanceConfig, IndexConfig, IndexAndQueryParam, InitialExploreParamsConfig


class Config(BaseModel):
    """纯ORM风格配置管理类"""
    output_dir: str = Field(default='./results')
    reports_dir: str = Field(default='./reports')
    connection: ConnectionConfig = Field(default_factory=ConnectionConfig)
    table_info: TableInfoConfig = Field(default_factory=TableInfoConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    index_config: IndexConfig = Field(default_factory=IndexConfig)
    initial_explore_params: InitialExploreParamsConfig = Field(default_factory=InitialExploreParamsConfig)
    manual_index_params: List[Dict[str, Any]] = []
    parallel_workers: Optional[int] = None
    explore_times: int = 50
    
    @field_validator('parallel_workers', mode='before')
    @classmethod
    def set_parallel_workers(cls, v):
        if v is None:
            return max(1, int(multiprocessing.cpu_count() * 0.75))
        return v
    
    @field_validator('explore_times', mode='before')
    @classmethod
    def set_explore_times(cls, v):
        if v is None:
            return 20
        return v

    def prepare_with_yaml(self, config_path) -> 'Config':
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件 {config_path} 不存在")
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                logger.info(f"成功加载配置文件: {config_path}")

                new_config = self.model_validate(config_dict)

                for field_name, field_value in new_config.model_dump().items():
                    if hasattr(self, field_name):
                        setattr(self, field_name, field_value)
                return self
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    def to_yaml(self, config_path: str) -> None:
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
