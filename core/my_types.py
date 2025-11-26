from typing import Dict, List, Optional, Any
from pydantic import BaseModel, field_validator


class ConnectionConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 6432
    user: str = "ann"
    password: str = "Huawei@123"
    dbname: str = "ann"


class TableInfoConfig(BaseModel):
    table_name: str = "items"
    vector_column_name: str = "embedding"
    dimension: int = 128
    metric: str = "l2" # l2, ip, cosine
    _sample_table_name: Optional[str] = None
    sample_table_count: int = 0
    _query_table_name: Optional[str] = None
    query_table_count: int = 0
    original_table_count: int = 0


    @field_validator('metric')
    @classmethod
    def validate_metric(cls, v):
        valid_metrics = ['l2', 'ip', 'cosine']
        if v not in valid_metrics:
            raise ValueError(f"metric必须是以下之一: {', '.join(valid_metrics)}")
        return v

    @property
    def sample_table_name(self) -> str:
        if self._sample_table_name is None:
            self._sample_table_name = f"{self.table_name}_sample_vecindex_finder"
        return self._sample_table_name

    @property
    def query_table_name(self) -> str:
        if self._query_table_name is None:
            self._query_table_name = f"{self.table_name}_query_vecindex_finder"
        return self._query_table_name


class SamplingConfig(BaseModel):
    default_ratio: float = 0.1
    min_sample_count: int = 10000
    max_sample_count: int = 10000000


class QueryConfig(BaseModel):
    query_get_type: str = "sample"
    query_count: int = 1000
    query_data_path: str = "query_data.json"


class PerformanceConfig(BaseModel):
    limit: int = 100
    min_recall: float = 0.95
    weight: Dict[str, float] = {
        "create_index_time": 0.03,
        "index_size": 0.12,
        "qps": 0.85
    }



class InitialExploreParamsConfig(BaseModel):
    manual_param: bool = False
    index_param: Dict[str, Any] = {}
    query_param: Dict[str, Any] = {}



class IndexConfig(BaseModel):
    find_index_type: str = "graph_index"
    auto: bool = True
    prepare_cache: bool = True

    @field_validator('find_index_type')
    @classmethod
    def validate_index_type(cls, v):
        valid_types = ['graph_index', 'diskann', 'ivfflat', 'ivfpq']
        if v not in valid_types:
            raise ValueError(f"索引类型必须是以下之一: {', '.join(valid_types)}")
        return v


class QueryData(BaseModel):
    id: int
    vectors: List[Any]
    distances: List[Any]
    

class IndexAndQueryParam(BaseModel):
    index_type: str
    index_param: Optional[Dict[str, Any]] = None
    query_param: Dict[str, Any]

    def to_dict(self):
        return {
            "index_type": self.index_type,
            "index_param": self.index_param,
            "query_param": self.query_param
        }
