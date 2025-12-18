from core.engine import db_engine, DatabaseEngine
from typing import Optional

MODULE_MAP = {
    "ivfflat": "IvfflatModule",
    "ivfpq": "IvfpqModule",
    "hnsw": "HnswModule",
    "graph_index": "HnswModule",
    "diskann": "DiskannModule",
}

class BaseModule(object):
    def __init__(self, table_name, vector_column_name, metric, parallel_workers=8, db_engine_obj: Optional[DatabaseEngine] = None):
        self.db_engine = db_engine_obj or db_engine
        self.conn = self.db_engine.get_connection()
        self._metric = metric
        self._cur = self.conn.cursor()
        self.parallel_workers = parallel_workers
        self.tablename = table_name
        self.vector_column_name = vector_column_name
        opr = ''
        if metric == "angular" or metric == "cosine":
            opr = '<=>'
        elif metric == "euclidean" or metric == "l2":
            opr = '<->'
        elif metric == "ip" or metric == "inner_product":
            opr = '<+>'
        else:
            raise RuntimeError(f"unknown metric {metric}")
        self._query = f"SELECT /*+ use_gplan indexscan({self.tablename}) */ {self.vector_column_name} {opr} %s as distance FROM {self.tablename} ORDER BY distance LIMIT %s"

    def done(self):
        self.conn.close()

    def create_index(self):
        raise NotImplementedError("create_index method must be implemented")

    def set_query_arguments(self, **kwargs):
        raise NotImplementedError("set_query_arguments method must be implemented")

    def preprocess_query(self, v):
        return "[" + ", ".join(f"{x:.6g}" for x in v) + "]"

    def query(self, v_str, n):
        self._cur.execute(self._query, (v_str, n))
        res = [distance for distance, in self._cur.fetchall()]
        return res

    def get_index_usage(self):
        self._cur.execute(f"SELECT pg_table_size('{self.tablename}_{self.vector_column_name}_idx')")
        return self._cur.fetchone()[0] / (1024 * 1024)

    def get_table_usage(self):
        self._cur.execute(f"SELECT pg_table_size('{self.tablename}')")
        return self._cur.fetchone()[0] / (1024 * 1024)

    def __str__(self):
        raise NotImplementedError("__str__ method must be implemented")

class IvfflatModule(BaseModule):
    def __init__(self, table_name, vector_column_name, ivf_nlist, metric, parallel_workers=8, db_engine_obj: Optional[DatabaseEngine] = None):
        super().__init__(table_name, vector_column_name, metric, parallel_workers, db_engine_obj)
        self._n_list = ivf_nlist
        self._n_probe = None

    def create_index(self):
        """Create index"""
        # 创建索引前，先删除索引
        self._cur.execute(f"DROP INDEX IF EXISTS {self.tablename}_{self.vector_column_name}_idx;")
        print("creating index...")
        ops = ''
        if self._metric == "angular" or self._metric == "cosine":
            ops = 'floatvector_cosine_ops'
        elif self._metric == "euclidean" or self._metric == "l2":
            ops = 'floatvector_l2_ops'
        elif self._metric == "ip" or self._metric == "inner_product":
            ops = 'floatvector_ip_ops'
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        self._cur.execute(
            f"CREATE INDEX ON {self.tablename} USING ivfflat ({self.vector_column_name} {ops}) WITH (ivf_nlist = {self._n_list}, parallel_workers = {self.parallel_workers})"
        )
        print("done!")

    def set_query_arguments(self, **kwargs):
        # 通过关键字参数接收ivf_probes
        if 'ivf_probes' in kwargs:
            self._n_probe = kwargs['ivf_probes']
            self._cur.execute("set ivf_probes = %d" % self._n_probe)
        else:
            raise ValueError("缺少必要的参数'ivf_probes'")

    def __str__(self):
        return "ivfflat(n_list=%d, n_probe=%d)" % (self._n_list, self._n_probe)


class IvfpqModule(BaseModule):
    def __init__(self, table_name, vector_column_name, ivf_nlist, metric, num_subquantizers=32, nbits=8, parallel_workers=8, db_engine_obj: Optional[DatabaseEngine] = None):
        super().__init__(table_name, vector_column_name, metric, parallel_workers, db_engine_obj)
        self._m = num_subquantizers  # 默认值
        self._n_list = ivf_nlist
        self._nbits = nbits  # 默认值
        self._n_probe = None
        self._n_factor = None

    def create_index(self):
        """Override create_index for IVF-PQ"""
        self._cur.execute(f"DROP INDEX IF EXISTS {self.tablename}_{self.vector_column_name}_idx;")
        print("creating pq index...")
        ops = ''
        if self._metric == "angular" or self._metric == "cosine":
            ops = 'floatvector_cosine_ops'
        elif self._metric == "euclidean" or self._metric == "l2":
            ops = 'floatvector_l2_ops'
        elif self._metric == "ip" or self._metric == "inner_product":
            ops = 'floatvector_ip_ops'
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        self._cur.execute(
            f"CREATE INDEX ON {self.tablename} USING ivfpq ({self.vector_column_name} {ops}) WITH (ivf_nlist = {self._n_list}, num_subquantizers = {self._m}, nbits = {self._nbits}, parallel_workers = {self.parallel_workers})"
        )
        print("done!")

    def set_query_arguments(self, **kwargs):
        # 通过关键字参数接收ivf_probes和ivfpq_refine_k_factor
        if 'ivf_probes' in kwargs and 'ivfpq_refine_k_factor' in kwargs:
            self._n_probe = kwargs['ivf_probes']
            self._n_factor = kwargs['ivfpq_refine_k_factor']
            self._cur.execute(f"set ivf_probes = {self._n_probe}; set ivfpq_refine_k_factor={self._n_factor};")
        else:
            missing_params = []
            if 'ivf_probes' not in kwargs:
                missing_params.append('ivf_probes')
            if 'ivfpq_refine_k_factor' not in kwargs:
                missing_params.append('ivfpq_refine_k_factor')
            raise ValueError(f"缺少必要的参数: {', '.join(missing_params)}")

    def __str__(self):
        return "ivfpq(n_list=%d, n_probe=%d, ivfpq_refine_k_factor=%d, num_subquantizers=%d, nbits=%d)" % (self._n_list, self._n_probe, self._n_factor, self._m, self._nbits)


class HnswModule(BaseModule):
    def __init__(self, table_name, vector_column_name, metric, m, ef_construction, quantizer='\"none\"', parallel_workers=8, db_engine_obj: Optional[DatabaseEngine] = None):
        super().__init__(table_name, vector_column_name, metric, parallel_workers, db_engine_obj)
        self._m = m
        self._ef_construction = ef_construction
        self._ef_search = None
        self._quantizer = quantizer

    def create_index(self):
        """Override create_index for HNSW"""
        self._cur.execute(f"DROP INDEX IF EXISTS {self.tablename}_{self.vector_column_name}_idx;")
        print("creating index...")
        metric_str = ''
        if self._metric == "angular" or self._metric == "cosine":
            metric_str = 'floatvector_cosine_ops'
        elif self._metric == "euclidean" or self._metric == "l2":
            metric_str = 'floatvector_l2_ops'
        elif self._metric == "ip" or self._metric == "inner_product":
            metric_str = 'floatvector_ip_ops'
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        self._cur.execute(
            f"CREATE INDEX ON {self.tablename} USING hnsw ({self.vector_column_name} {metric_str}) WITH (m={self._m}, ef_construction={self._ef_construction}, quantizer={self._quantizer}, parallel_workers={self.parallel_workers})"
        )
        print("done!")

    def set_query_arguments(self, **kwargs):
        # 通过关键字参数接收hnsw_ef_search
        if 'hnsw_ef_search' in kwargs:
            self._ef_search = kwargs['hnsw_ef_search']
            self._cur.execute("set hnsw_ef_search = %d" % self._ef_search)
        else:
            raise ValueError("缺少必要的参数'hnsw_ef_search'")

    def __str__(self):
        return f"graph_index(m={self._m}, ef_construction={self._ef_construction}, quantizer={self._quantizer}, hnsw_ef_search={self._ef_search})"


class DiskannModule(BaseModule):
    def __init__(self, table_name, vector_column_name, metric, m=99, ef_construction=120, occlusion_factor=1.2, parallel_workers=8, db_engine_obj: Optional[DatabaseEngine] = None):
        super().__init__(table_name, vector_column_name, metric, parallel_workers, db_engine_obj)
        self._ef_search = None
        self._m = m
        self._ef_construction = ef_construction
        self._occlusion_factor = occlusion_factor
        
    def create_index(self):
        self._cur.execute(f"DROP INDEX IF EXISTS {self.tablename}_{self.vector_column_name}_idx;")
        print("creating index...")
        ops = ''
        if self._metric == "angular" or self._metric == "cosine":
            ops = 'floatvector_cosine_ops'
        elif self._metric == "euclidean" or self._metric == "l2":
            ops = 'floatvector_l2_ops'
        elif self._metric == "ip" or self._metric == "inner_product":
            ops = 'floatvector_ip_ops'
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        self._cur.execute(
            f"CREATE INDEX ON {self.tablename} USING diskann ({self.vector_column_name} {ops}) WITH (parallel_workers = {self.parallel_workers}, m={self._m}, ef_construction={self._ef_construction}, occlusion_factor={self._occlusion_factor});"
        )
        print("done!")

    def set_query_arguments(self, **kwargs):
        # 通过关键字参数接收diskann_search_list_size
        if 'diskann_search_list_size' in kwargs:
            self._ef_search = kwargs['diskann_search_list_size']
            self._cur.execute("set diskann_search_list_size = %d;" % self._ef_search)
        else:
            raise ValueError("缺少必要的参数'diskann_search_list_size'")

    def __str__(self):
        return f"diskann(m={self._m}, ef_construction={self._ef_construction}, occlusion_factor={self._occlusion_factor}, diskann_search_list_size={self._ef_search})"
