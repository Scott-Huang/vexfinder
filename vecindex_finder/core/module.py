from core.engine import db_engine, DatabaseEngine
from typing import Optional

MODULE_MAP = {
    "ivfflat": "IvfflatModule",
    "ivfpq": "IvfpqModule",
    "hnsw": "HnswModule",
    "diskann": "DiskannModule",
}

class BaseModule(object):
    def __init__(self, table_name, vector_column_name, metric, parallel_workers=8, db_engine_obj: Optional[DatabaseEngine] = None):
        self.db_engine = db_engine_obj or db_engine
        self.conn = self.db_engine.get_connection()
        self._metric = metric
        self._cur = self.conn.cursor()
        self.parallel_workers = parallel_workers  # 默认值
        self.tablename = table_name
        self.vector_column_name = vector_column_name
        self._query = None

        if metric == "angular" or metric == "cosine":
            self._query = f"SELECT {self.vector_column_name} <=> %s as distance FROM {self.tablename} ORDER BY distance LIMIT %s"
        elif metric == "euclidean" or metric == "l2":
            self._query = f"SELECT {self.vector_column_name} <-> %s as distance FROM {self.tablename} ORDER BY distance LIMIT %s"
        elif metric == "ip" or metric == "inner_product":
            self._query = f"SELECT {self.vector_column_name} <-> %s as distance FROM {self.tablename} ORDER BY distance LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")

    def done(self):
        self.conn.close()

    def create_index(self):
        raise NotImplementedError("create_index method must be implemented")

    def set_query_arguments(self, **kwargs):
        raise NotImplementedError("set_query_arguments method must be implemented")

    def query(self, v, n):
        v_str = "[" + ", ".join(f"{x:.8f}" for x in v) + "]"
        self._cur.execute(self._query, (v_str, n))
        res = [distance for distance, in self._cur.fetchall()]
        return res

    def get_memory_usage(self):
        if self._cur is None:
            return 0
        self._cur.execute(f"SELECT pg_relation_size('{self.tablename}_{self.vector_column_name}_idx')")
        return self._cur.fetchone()[0] / 1024

    def get_table_usage(self):
        self._cur.execute(f"SELECT pg_relation_size('{self.tablename}')")
        return self._cur.fetchone()[0] / 1024

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
        if self._metric == "angular" or self._metric == "cosine":
            self._cur.execute(
                f"CREATE INDEX ON {self.tablename} USING ivfflat ({self.vector_column_name} floatvector_cosine_ops) WITH (ivf_nlist = {self._n_list})"
            )
        elif self._metric == "euclidean" or self._metric == "l2":
            self._cur.execute(
                f"CREATE INDEX ON {self.tablename} USING ivfflat ({self.vector_column_name} floatvector_l2_ops) WITH (ivf_nlist = {self._n_list})"
            )
        elif self._metric == "ip" or self._metric == "inner_product":
            self._cur.execute(
                f"CREATE INDEX ON {self.tablename} USING ivfflat ({self.vector_column_name} floatvector_ip_ops) WITH (ivf_nlist = {self._n_list})"
            )
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        print("done!")

    def set_query_arguments(self, ivf_probes):
        self._n_probe = ivf_probes
        self._cur.execute("SET enable_seqscan TO off; set ivf_probes = %d" % ivf_probes)

    def __str__(self):
        return "ivfflat-index(n_list=%d, n_probe=%d)" % (self._n_list, self._n_probe)


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
        if self._metric == "angular" or self._metric == "cosine":
            self._cur.execute(
                f"CREATE INDEX ON {self.tablename} USING ivfpq ({self.vector_column_name} floatvector_cosine_ops) WITH (ivf_nlist = {self._n_list}, num_subquantizers = {self._m}, nbits = {self._nbits}, parallel_workers = {self.parallel_workers})"
            )
        elif self._metric == "euclidean" or self._metric == "l2":
            self._cur.execute(
                f"CREATE INDEX ON {self.tablename} USING ivfpq ({self.vector_column_name} floatvector_l2_ops) WITH (ivf_nlist = {self._n_list}, num_subquantizers = {self._m}, nbits = {self._nbits}, parallel_workers = {self.parallel_workers})"
            )
        elif self._metric == "ip" or self._metric == "inner_product":
            self._cur.execute(
                f"CREATE INDEX ON {self.tablename} USING ivfpq ({self.vector_column_name} floatvector_ip_ops) WITH (ivf_nlist = {self._n_list}, num_subquantizers = {self._m}, nbits = {self._nbits}, parallel_workers = {self.parallel_workers})"
            )
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        print("done!")

    def set_query_arguments(self, ivf_probes, ivfpq_refine_k_factor):
        self._n_probe = ivf_probes
        self._n_factor = ivfpq_refine_k_factor
        self._cur.execute(
            f"SET enable_seqscan TO off; set ivf_probes = {ivf_probes}; set ivfpq_refine_k_factor={ivfpq_refine_k_factor};")

    def __str__(self):
        return "ivfpq-index(n_list=%d, n_probe=%d, ivfpq_refine_k_factor=%d, num_subquantizers=%d, nbits=%d)" % (self._n_list, self._n_probe, self._n_factor, self._m, self._nbits)


class HnswModule(BaseModule):
    def __init__(self, table_name, vector_column_name, metric, m, ef_construction, parallel_workers=8, db_engine_obj: Optional[DatabaseEngine] = None):
        super().__init__(table_name, vector_column_name, metric, parallel_workers, db_engine_obj)
        self._m = m
        self._ef_construction = ef_construction
        self._ef_search = None

    def create_index(self):
        """Override create_index for HNSW"""
        self._cur.execute(f"DROP INDEX IF EXISTS {self.tablename}_{self.vector_column_name}_idx;")
        print("creating index...")
        if self._metric == "angular" or self._metric == "cosine":
            self._cur.execute(
                f"CREATE INDEX ON {self.tablename} USING hnsw ({self.vector_column_name} floatvector_cosine_ops) WITH (m = {self._m}, ef_construction = {self._ef_construction}, parallel_workers={self.parallel_workers})"
            )
        elif self._metric == "euclidean" or self._metric == "l2":
            self._cur.execute(
                f"CREATE INDEX ON {self.tablename} USING hnsw ({self.vector_column_name} floatvector_l2_ops) WITH (m = {self._m}, ef_construction = {self._ef_construction}, parallel_workers={self.parallel_workers})"
            )
        elif self._metric == "ip" or self._metric == "inner_product":
            self._cur.execute(
                f"CREATE INDEX ON {self.tablename} USING hnsw ({self.vector_column_name} floatvector_ip_ops) WITH (m = {self._m}, ef_construction = {self._ef_construction}, parallel_workers={self.parallel_workers})"
            )
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        print("done!")

    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search
        self._cur.execute(
            "SET enable_seqscan TO off; set hnsw_ef_search = %d" % ef_search)

    def __str__(self):
        return f"hnsw-index(m={self._m}, ef_construction={self._ef_construction}, hnsw_ef_search={self._ef_search})"


class DiskannModule(BaseModule):
    def __init__(self, table_name, vector_column_name, metric, parallel_workers=8, db_engine_obj: Optional[DatabaseEngine] = None):
        super().__init__(table_name, vector_column_name, metric, parallel_workers, db_engine_obj)

    def create_index(self):
        self._cur.execute(f"DROP INDEX IF EXISTS {self.tablename}_{self.vector_column_name}_idx;")
        print("creating index...")
        if self._metric == "angular" or self._metric == "cosine":
            self._cur.execute(
                f"CREATE INDEX ON {self.tablename} USING diskann ({self.vector_column_name} floatvector_cosine_ops) WITH (parallel_workers = {self.parallel_workers}, enable_quantization=off);")
        elif self._metric == "euclidean" or self._metric == "l2":
            self._cur.execute(
                f"CREATE INDEX ON {self.tablename} USING diskann ({self.vector_column_name} floatvector_l2_ops) WITH (parallel_workers = {self.parallel_workers}, enable_quantization=off);")
        elif self._metric == "ip" or self._metric == "inner_product":
            self._cur.execute(
                f"CREATE INDEX ON {self.tablename} USING diskann ({self.vector_column_name} floatvector_ip_ops) WITH (parallel_workers = {self.parallel_workers}, enable_quantization=off);")
        else:
            raise RuntimeError(f"unknown metric {self._metric}")

    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search
        self._cur.execute(
            "SET enable_seqscan TO off; set diskann_search_list_size = %d; set diskann_query_with_pq = off;" % ef_search)

    def __str__(self):
        return f"diskann-index(diskann_search_list_size={self._ef_search})"
