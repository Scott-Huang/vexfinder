#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
向量索引参数优化工具 - 纯ORM风格配置类（不兼容旧代码）
"""

import os
import yaml
import multiprocessing
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, validator, root_validator


class ConnectionConfig(BaseModel):
    """数据库连接配置"""
    host: str = "127.0.0.1"
    port: int = 6432
    user: str = "ann"
    password: str = "Huawei@123"
    dbname: str = "ann"


class TableInfoConfig(BaseModel):
    """表信息配置"""
    table_name: str = "items"
    vector_column_name: str = "embedding"
    dimension: int = 128
    metric: str = "l2"
    sample_table_name: Optional[str] = None
    query_table_name: Optional[str] = None
    
    @validator('sample_table_name', 'query_table_name', pre=True, always=True)
    def set_table_names(cls, v, values):
        """设置表名"""
        if v is None:
            table_name = values.get('table_name', 'items')
            if v is values.get('sample_table_name'):
                return f"{table_name}_sample_vecindex_finder"
            else:
                return f"{table_name}_query_vecindex_finder"
        return v


class SamplingConfig(BaseModel):
    """采样配置"""
    default_ratio: float = 0.1
    min_sample_count: int = 10000
    max_sample_count: int = 1000000


class QueryConfig(BaseModel):
    """查询配置"""
    query_get_type: str = "sample"
    query_count: int = 1000
    query_data_path: str = "query_data.json"


class PerformanceConfig(BaseModel):
    """性能测试配置"""
    limit: int = 100
    min_recall: float = 0.9
    weight: Dict[str, float] = {
        "create_index_time": 0.3,
        "index_size": 0.1,
        "qps": 0.6
    }


class IvfFlatConfig(BaseModel):
    """IVF-FLAT索引参数"""
    nlist: List[int] = [16, 32, 64, 128, 256, 512, 1024]
    nprobe: List[int] = [1, 4, 8, 16, 32, 64, 128]


class IvfPqConfig(BaseModel):
    """IVF-PQ索引参数"""
    nlist: List[int] = [16, 32, 64, 128, 256, 512, 1024]
    m: List[int] = [4, 8, 16, 32]
    nbits: int = 8
    nprobe: List[int] = [1, 4, 8, 16, 32, 64]
    ivfpq_refine_k_factor: List[int] = [1, 2, 4, 8]


class HnswConfig(BaseModel):
    """HNSW索引参数"""
    M: List[int] = [8, 16, 32, 64]
    efConstruction: List[int] = [40, 80, 120, 200, 400]
    ef: List[int] = [10, 20, 40, 80, 100, 200, 400, 800]


class IndexConfig(BaseModel):
    """索引配置"""
    find_index_type: str = "ivfflat"
    auto: bool = True
    ivfflat: IvfFlatConfig = Field(default_factory=IvfFlatConfig)
    ivfpq: IvfPqConfig = Field(default_factory=IvfPqConfig)
    hnsw: HnswConfig = Field(default_factory=HnswConfig)
    
    @validator('find_index_type')
    def validate_index_type(cls, v):
        """验证索引类型"""
        valid_types = ['ivfflat', 'ivfpq', 'hnsw']
        if v not in valid_types:
            raise ValueError(f"索引类型必须是以下之一: {', '.join(valid_types)}")
        return v


class Config(BaseModel):
    """纯ORM风格配置管理类"""
    connection: ConnectionConfig = Field(default_factory=ConnectionConfig)
    table_info: TableInfoConfig = Field(default_factory=TableInfoConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    parallel_workers: Optional[int] = None
    
    @validator('parallel_workers', pre=True, always=True)
    def set_parallel_workers(cls, v):
        """设置并行工作线程数"""
        if v is None:
            return max(1, int(multiprocessing.cpu_count() * 0.75))
        return v
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """从YAML文件加载配置"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件 {config_path} 不存在")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.parse_obj(config_dict)
    
    def to_yaml(self, config_path: str) -> None:
        """将配置保存为YAML文件"""
        with open(config_path, 'w') as f:
            yaml.dump(self.dict(exclude_none=True), f, default_flow_style=False)
    
    class Config:
        """Pydantic配置类"""
        # 允许额外属性
        extra = "ignore"
        # 验证赋值
        validate_assignment = True
        # 允许使用任意类型
        arbitrary_types_allowed = True


# 使用示例
def main():
    """配置使用示例"""
    # 创建默认配置
    config = Config()
    print(f"默认配置的表名: {config.table_info.table_name}")
    print(f"默认配置的采样表名: {config.table_info.sample_table_name}")
    
    # 修改配置
    config.table_info.table_name = "vectors"
    config.index.find_index_type = "hnsw"
    config.index.hnsw.M = [16, 32]
    
    print(f"修改后的表名: {config.table_info.table_name}")
    print(f"修改后的采样表名: {config.table_info.sample_table_name}")
    print(f"修改后的索引类型: {config.index.find_index_type}")
    print(f"修改后的HNSW M参数: {config.index.hnsw.M}")
    
    # 保存配置到YAML文件
    config.to_yaml("new_config.yml")
    
    # 从YAML文件加载配置
    loaded_config = Config.from_yaml("new_config.yml")
    print(f"加载的配置表名: {loaded_config.table_info.table_name}")
    print(f"加载的配置索引类型: {loaded_config.index.find_index_type}")
    
    # 环境变量和命令行参数支持示例
    # 可以通过环境变量设置配置项
    import os
    os.environ["VECINDEX_CONNECTION_HOST"] = "192.168.1.100"
    
    # 从环境变量加载配置
    from pydantic import BaseSettings
    
    class EnvConfig(BaseSettings):
        """支持环境变量的配置类"""
        connection: ConnectionConfig = Field(default_factory=ConnectionConfig)
        
        class Config:
            env_prefix = "VECINDEX_"
            env_nested_delimiter = "_"
    
    env_config = EnvConfig()
    print(f"从环境变量加载的主机: {env_config.connection.host}")


if __name__ == "__main__":
    main() 