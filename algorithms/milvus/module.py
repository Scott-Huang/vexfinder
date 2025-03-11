from time import sleep
from pymilvus import DataType, connections, utility, Collection, CollectionSchema, FieldSchema, DataType
import os

from ..base.module import BaseANN


def metric_mapping(_metric: str):
    _metric_type = {"angular": "COSINE", "euclidean": "L2"}.get(_metric, None)
    if _metric_type is None:
        raise Exception(f"[Milvus] Not support metric type: {_metric}!!!")
    return _metric_type


def get_conn():
    conn = connections.connect("default", host='localhost', port='19530')
    return conn


class Milvus(BaseANN):
    def __init__(self, metric, dim, index_param):
        self._metric = metric
        self._dim = dim
        self._metric_type = metric_mapping(self._metric)
        self.connects = get_conn()
        self.collection_name = "test_milvus"
        self.collection = None

    def done(self):
        self.drop()

    def drop(self):
        """Drop existing collection"""
        if utility.has_collection(self.collection_name):
            print(f"[Milvus] collection {self.collection_name} already exists, drop it...")
            utility.drop_collection(self.collection_name)

    def create_collection(self):
        filed_id = FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True
        )
        filed_vec = FieldSchema(
            name="vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=self._dim
        )
        schema = CollectionSchema(
            fields=[filed_id, filed_vec],
            description="Test milvus search",
        )
        self.collection = Collection(
            self.collection_name,
            schema,
            consistence_level="STRONG"
        )
        print(f"[Milvus] Create collection {self.collection.describe()} successfully!!!")

    def copy(self, dataset):
        """Copy data to table using batch processing
        Args:
            dataset: h5py Dataset object
        """
        # 创建集合
        self.create_collection()
        
        # 获取训练数据集
        train_data = dataset["train"]
        print(f"[Milvus] Insert {train_data.shape[0]} data into collection {self.collection_name}...")
        
        # 批量处理
        batch_size = 10000
        total_rows = train_data.shape[0]
        
        print(f"[Milvus] Start copying {total_rows} rows with batch size {batch_size}...")
        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            batch = train_data[start_idx:end_idx]  # 只读取一部分数据
            
            # Convert batch data to numpy array directly
            entities = [
                [i for i in range(start_idx, end_idx)],  # id list
                batch.tolist()  # vector data
            ]
            self.collection.insert(entities)
            
            if end_idx % (total_rows // 10) == 0 or end_idx == total_rows:
                print(f"[Milvus] Processed {end_idx}/{total_rows} rows ({(end_idx/total_rows*100):.2f}%)")
        
        self.collection.flush()
        print(f"[Milvus] {self.collection.num_entities} data has been inserted into collection {self.collection_name}!!!")

    def get_index_param(self):
        raise NotImplementedError()

    def create_index(self):
        # create index
        if not self.collection:
            self.collection = Collection(
                self.collection_name,
                consistence_level="STRONG"
            )
        print(f"[Milvus] Create index for collection {self.collection_name}...")
        self.collection.create_index(
            field_name="vector",
            index_params=self.get_index_param(),
            index_name="vector_index"
        )
        utility.wait_for_index_building_complete(
            collection_name=self.collection_name,
            index_name="vector_index"
        )
        index = self.collection.index(index_name="vector_index")
        index_progress = utility.index_building_progress(
            collection_name=self.collection_name,
            index_name="vector_index"
        )
        print(
            f"[Milvus] Create index {index.to_dict()} {index_progress} for collection {self.collection_name} successfully!!!")
        self.load_collection()

    def load_collection(self):
        # load collection
        print(f"[Milvus] Load collection {self.collection_name}...")
        self.collection.load()
        utility.wait_for_loading_complete(self.collection_name)
        print(f"[Milvus] Load collection {self.collection_name} successfully!!!")

    def query(self, v, n):
        if not self.collection:
            self.collection = Collection(
                self.collection_name,
                consistence_level="STRONG"
            )
        results = self.collection.search(
            data=[v],
            anns_field="vector",
            param=self.search_params,
            limit=n,
            output_fields=["id"]
        )
        ids = [r.entity.get("id") for r in results[0]]
        return ids


class MilvusFLAT(Milvus):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self.name = f"MilvusFLAT metric:{self._metric}"

    def get_index_param(self):
        return {
            "index_type": "FLAT",
            "metric_type": self._metric_type
        }

    def query(self, v, n):
        results = self.collection.search(
            data=[v],
            anns_field="vector",
            param=self.search_params,
            limit=n,
            output_fields=["id"]
        )
        ids = [r.entity.get("id") for r in results[0]]
        return ids

    def copy(self, X):
        self.create_collection()
        print(f"[Milvus] Insert {len(X)} data into collection {self.collection_name}...")
        batch_size = 1000
        for i in range(0, len(X), batch_size):
            batch_data = X[i: min(i + batch_size, len(X))]
            entities = [
                [i for i in range(i, min(i + batch_size, len(X)))],
                batch_data.tolist()
            ]

    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self._index_nlist = index_param.get("nlist", None)

    def get_index_param(self):
        return {
            "index_type": "IVF_FLAT",
            "params": {
                "nlist": self._index_nlist
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, nprobe):
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"milvus-ivfflat metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"


class MilvusIVFSQ8(Milvus):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self._index_nlist = index_param.get("nlist", None)

    def get_index_param(self):
        return {
            "index_type": "IVF_SQ8",
            "params": {
                "nlist": self._index_nlist
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, nprobe):
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"milvus-ivfsq8 metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"


class MilvusIVFFLAT(Milvus):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self._index_nlist = index_param.get("nlist", None)

    def get_index_param(self):
        return {
            "index_type": "IVF_FLAT",
            "params": {
                "nlist": self._index_nlist,
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, nprobe):
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"milvus-ivfflat metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"


class MilvusIVFPQ(Milvus):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self._index_nlist = index_param.get("nlist", None)
        self._index_m = index_param.get("m", None)
        self._index_nbits = index_param.get("nbits", None)

    def get_index_param(self):
        assert self._dim % self._index_m == 0, "dimension must be able to be divided by m"
        return {
            "index_type": "IVF_PQ",
            "params": {
                "nlist": self._index_nlist,
                "m": self._index_m,
                "nbits": self._index_nbits if self._index_nbits else 8
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, nprobe):
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"milvus-ivfpq metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"


class MilvusHNSW(Milvus):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self._index_m = index_param.get("M", None)
        self._index_ef = index_param.get("efConstruction", None)

    def get_index_param(self):
        return {
            "index_type": "HNSW",
            "params": {
                "M": self._index_m,
                "efConstruction": self._index_ef
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, ef):
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"ef": ef}
        }
        self.name = f"milvus-hnsw metric:{self._metric}, index_M:{self._index_m}, index_ef:{self._index_ef}, search_ef={ef}"


class MilvusSCANN(Milvus):
    def __init__(self, metric, dim, index_param):
        super().__init__(metric, dim, index_param)
        self._index_nlist = index_param.get("nlist", None)

    def get_index_param(self):
        return {
            "index_type": "SCANN",
            "params": {
                "nlist": self._index_nlist
            },
            "metric_type": self._metric_type
        }

    def set_query_arguments(self, nprobe):
        self.search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": nprobe}
        }
        self.name = f"milvus-scann metric:{self._metric}, index_nlist:{self._index_nlist}, search_nprobe:{nprobe}"
