#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
向量索引参数优化工具 - ORM风格配置类示例
"""

import os
import yaml
import multiprocessing
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from pydantic import BaseModel, Field, validator

# 方案一：使用dataclasses实现ORM风格配置

@dataclass
class ConnectionConfig:
    """数据库连接配置"""
    host: str = "127.0.0.1"
    port: int = 6432
    user: str = "ann"
    password: str = "Huawei@123"
    dbname: str = "ann"


@dataclass
class TableInfoConfig:
    """表信息配置"""
    table_name: str = "items"
    vector_column_name: str = "embedding"
    dimension: int = 128
    metric: str = "l2"
    sample_table_name: str = None
    query_table_name: str = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.sample_table_name is None:
            self.sample_table_name = f"{self.table_name}_sample_vecindex_finder"
        if self.query_table_name is None:
            self.query_table_name = f"{self.table_name}_query_vecindex_finder"


@dataclass
class SamplingConfig:
    """采样配置"""
    default_ratio: float = 0.1
    min_sample_count: int = 10000
    max_sample_count: int = 1000000


@dataclass
class QueryConfig:
    """查询配置"""
    query_get_type: str = "sample"
    query_count: int = 1000
    query_data_path: str = "query_data.json"


@dataclass
class PerformanceConfig:
    """性能测试配置"""
    limit: int = 100
    min_recall: float = 0.8


@dataclass
class IvfFlatConfig:
    """IVF-FLAT索引参数"""
    nlist: List[int] = field(default_factory=lambda: [16, 32, 64, 128, 256, 512, 1024])
    nprobe: List[int] = field(default_factory=lambda: [1, 4, 8, 16, 32, 64, 128])


@dataclass
class IvfPqConfig:
    """IVF-PQ索引参数"""
    nlist: List[int] = field(default_factory=lambda: [16, 32, 64, 128, 256, 512, 1024])
    m: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    nbits: int = 8
    nprobe: List[int] = field(default_factory=lambda: [1, 4, 8, 16, 32, 64])
    ivfpq_refine_k_factor: List[int] = field(default_factory=lambda: [1, 2, 4, 8])


@dataclass
class HnswConfig:
    """HNSW索引参数"""
    M: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    efConstruction: List[int] = field(default_factory=lambda: [40, 80, 120, 200, 400])
    ef: List[int] = field(default_factory=lambda: [10, 20, 40, 80, 100, 200, 400, 800])


@dataclass
class IndexConfig:
    """索引配置"""
    find_index_type: str = "ivfflat"
    auto: bool = True
    ivfflat: IvfFlatConfig = field(default_factory=IvfFlatConfig)
    ivfpq: IvfPqConfig = field(default_factory=IvfPqConfig)
    hnsw: HnswConfig = field(default_factory=HnswConfig)


@dataclass
class Config:
    """ORM风格配置管理类"""
    connection: ConnectionConfig = field(default_factory=ConnectionConfig)
    table_info: TableInfoConfig = field(default_factory=TableInfoConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    parallel_workers: int = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.parallel_workers is None:
            self.parallel_workers = max(1, int(multiprocessing.cpu_count() * 0.75))
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """从YAML文件加载配置"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件 {config_path} 不存在")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # 创建配置对象
        connection = ConnectionConfig(**config_dict.get('connection', {}))
        table_info = TableInfoConfig(**config_dict.get('table_info', {}))
        sampling = SamplingConfig(**config_dict.get('sampling', {}))
        query = QueryConfig(**config_dict.get('query', {}))
        performance = PerformanceConfig(**config_dict.get('performance', {}))
        
        # 处理索引配置
        index_dict = config_dict.get('index', {})
        ivfflat = IvfFlatConfig(**index_dict.get('ivfflat', {}))
        ivfpq = IvfPqConfig(**index_dict.get('ivfpq', {}))
        hnsw = HnswConfig(**index_dict.get('hnsw', {}))
        
        index = IndexConfig(
            find_index_type=index_dict.get('find_index_type', 'ivfflat'),
            auto=index_dict.get('auto', True),
            ivfflat=ivfflat,
            ivfpq=ivfpq,
            hnsw=hnsw
        )
        
        return cls(
            connection=connection,
            table_info=table_info,
            sampling=sampling,
            query=query,
            performance=performance,
            index=index,
            parallel_workers=config_dict.get('parallel_workers')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return asdict(self)
    
    def to_yaml(self, config_path: str) -> None:
        """将配置保存为YAML文件"""
        with open(config_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


# 方案二：使用Pydantic实现ORM风格配置

class PydanticConnectionConfig(BaseModel):
    """数据库连接配置"""
    host: str = "127.0.0.1"
    port: int = 6432
    user: str = "ann"
    password: str = "Huawei@123"
    dbname: str = "ann"


class PydanticTableInfoConfig(BaseModel):
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


class PydanticSamplingConfig(BaseModel):
    """采样配置"""
    default_ratio: float = 0.1
    min_sample_count: int = 10000
    max_sample_count: int = 1000000


class PydanticQueryConfig(BaseModel):
    """查询配置"""
    query_get_type: str = "sample"
    query_count: int = 1000
    query_data_path: str = "query_data.json"


class PydanticPerformanceConfig(BaseModel):
    """性能测试配置"""
    limit: int = 100
    min_recall: float = 0.9


class PydanticIvfFlatConfig(BaseModel):
    """IVF-FLAT索引参数"""
    nlist: List[int] = [16, 32, 64, 128, 256, 512, 1024]
    nprobe: List[int] = [1, 4, 8, 16, 32, 64, 128]


class PydanticIvfPqConfig(BaseModel):
    """IVF-PQ索引参数"""
    nlist: List[int] = [16, 32, 64, 128, 256, 512, 1024]
    m: List[int] = [4, 8, 16, 32]
    nbits: int = 8
    nprobe: List[int] = [1, 4, 8, 16, 32, 64]
    ivfpq_refine_k_factor: List[int] = [1, 2, 4, 8]


class PydanticHnswConfig(BaseModel):
    """HNSW索引参数"""
    M: List[int] = [8, 16, 32, 64]
    efConstruction: List[int] = [40, 80, 120, 200, 400]
    ef: List[int] = [10, 20, 40, 80, 100, 200, 400, 800]


class PydanticIndexConfig(BaseModel):
    """索引配置"""
    find_index_type: str = "ivfflat"
    auto: bool = True
    ivfflat: PydanticIvfFlatConfig = Field(default_factory=PydanticIvfFlatConfig)
    ivfpq: PydanticIvfPqConfig = Field(default_factory=PydanticIvfPqConfig)
    hnsw: PydanticHnswConfig = Field(default_factory=PydanticHnswConfig)


class PydanticConfig(BaseModel):
    """Pydantic风格配置管理类"""
    connection: PydanticConnectionConfig = Field(default_factory=PydanticConnectionConfig)
    table_info: PydanticTableInfoConfig = Field(default_factory=PydanticTableInfoConfig)
    sampling: PydanticSamplingConfig = Field(default_factory=PydanticSamplingConfig)
    query: PydanticQueryConfig = Field(default_factory=PydanticQueryConfig)
    performance: PydanticPerformanceConfig = Field(default_factory=PydanticPerformanceConfig)
    index: PydanticIndexConfig = Field(default_factory=PydanticIndexConfig)
    parallel_workers: Optional[int] = None
    
    @validator('parallel_workers', pre=True, always=True)
    def set_parallel_workers(cls, v):
        """设置并行工作线程数"""
        if v is None:
            return max(1, int(multiprocessing.cpu_count() * 0.75))
        return v
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'PydanticConfig':
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


# 使用示例
def dataclasses_example():
    """使用dataclasses实现的ORM风格配置示例"""
    # 创建默认配置
    config = Config()
    print(f"默认配置的表名: {config.table_info.table_name}")
    print(f"默认配置的采样表名: {config.table_info.sample_table_name}")
    
    # 修改配置
    config.table_info.table_name = "vectors"
    config.table_info.sample_table_name = f"{config.table_info.table_name}_sample"
    config.index.find_index_type = "hnsw"
    config.index.hnsw.M = [16, 32]
    
    print(f"修改后的表名: {config.table_info.table_name}")
    print(f"修改后的采样表名: {config.table_info.sample_table_name}")
    print(f"修改后的索引类型: {config.index.find_index_type}")
    print(f"修改后的HNSW M参数: {config.index.hnsw.M}")
    
    # 保存配置到YAML文件
    config.to_yaml("dataclasses_config.yml")
    
    # 从YAML文件加载配置
    loaded_config = Config.from_yaml("dataclasses_config.yml")
    print(f"加载的配置表名: {loaded_config.table_info.table_name}")
    print(f"加载的配置索引类型: {loaded_config.index.find_index_type}")


def pydantic_example():
    """使用Pydantic实现的ORM风格配置示例"""
    # 创建默认配置
    config = PydanticConfig()
    print(f"默认配置的表名: {config.table_info.table_name}")
    print(f"默认配置的采样表名: {config.table_info.sample_table_name}")
    
    # 修改配置
    config.table_info.table_name = "vectors"
    config.table_info.sample_table_name = f"{config.table_info.table_name}_sample"
    config.index.find_index_type = "hnsw"
    config.index.hnsw.M = [16, 32]
    
    print(f"修改后的表名: {config.table_info.table_name}")
    print(f"修改后的采样表名: {config.table_info.sample_table_name}")
    print(f"修改后的索引类型: {config.index.find_index_type}")
    print(f"修改后的HNSW M参数: {config.index.hnsw.M}")
    
    # 保存配置到YAML文件
    config.to_yaml("pydantic_config.yml")
    
    # 从YAML文件加载配置
    loaded_config = PydanticConfig.from_yaml("pydantic_config.yml")
    print(f"加载的配置表名: {loaded_config.table_info.table_name}")
    print(f"加载的配置索引类型: {loaded_config.index.find_index_type}")


if __name__ == "__main__":
    print("=== Dataclasses ORM风格配置示例 ===")
    dataclasses_example()
    
    print("\n=== Pydantic ORM风格配置示例 ===")
    pydantic_example() 