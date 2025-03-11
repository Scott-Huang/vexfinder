import subprocess
import sys
import psycopg2
import io
from ..base.module import BaseANN


def get_conn():
    conn = psycopg2.connect(host="127.0.0.1", user="ann", password="Huawei@123", dbname="ann", port=6432)
    conn.autocommit = True
    return conn

class OpenGaussIvfflatVector(BaseANN):
    def __init__(self, metric, nlist):
        super().__init__()
        self._n_list = nlist
        self._metric = metric
        self._cur = None
        self.parallel_workers = 8  # 默认值
        self.tablename = "items"
        self.conn = None
        self.conn = get_conn()
        self._cur = self.conn.cursor()
        self._n_probe = None

        if metric == "angular":
            self._query = f"SELECT id FROM {self.tablename} ORDER BY embedding <=> %s LIMIT %s"
        elif metric == "euclidean":
            self._query = f"SELECT id FROM {self.tablename} ORDER BY embedding <-> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")


    def done(self):
        self.drop()
        self.conn.close()

    def drop(self):
        """Drop existing table"""
        self._cur.execute(f"DROP TABLE IF EXISTS {self.tablename};")

    def copy(self, dataset):
        """Copy data to table using batch processing
        Args:
            dataset: h5py Dataset object
        """
        # 获取训练数据集
        train_data = dataset["train"]
        
        print("copying data...")
        # 创建表
        self._cur.execute(f"CREATE TABLE {self.tablename} (id int, embedding floatvector(({train_data.shape[1]})))")
        self._cur.execute(f"ALTER TABLE {self.tablename} ALTER COLUMN embedding SET STORAGE PLAIN")
        
        # 批量处理
        batch_size = 10000
        total_rows = train_data.shape[0]
        
        print(f"Start copying {total_rows} rows with batch size {batch_size}...")
        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            batch = train_data[start_idx:end_idx]  # 只读取一部分数据
            
            # 使用 StringIO 进行批量写入
            data = io.StringIO()
            for i, embedding in enumerate(batch):
                embedding_str = "[" + ", ".join(f"{x:.8f}".rstrip("0").rstrip(".") for x in embedding) + "]"
                data.write(f"{start_idx + i}\t{embedding_str}\n")
            
            data.seek(0)
            self._cur.copy_from(data, self.tablename, columns=('id', 'embedding'), sep='\t')
            
            if end_idx % (total_rows // 10) == 0 or end_idx == total_rows:
                print(f"Processed {end_idx}/{total_rows} rows ({(end_idx/total_rows*100):.2f}%)")

    def create_index(self):
        """Create index"""
        # 创建索引前，先删除索引
        self._cur.execute(f"DROP INDEX IF EXISTS {self.tablename}_embedding_idx;")
        print("creating index...")
        if self._metric == "angular":
            self._cur.execute(
                f"CREATE INDEX ON {self.tablename} USING ivfflat (embedding floatvector_cosine_ops) WITH (ivf_nlist = {self._n_list})"
            )
        elif self._metric == "euclidean":
            self._cur.execute(
                f"CREATE INDEX ON {self.tablename} USING ivfflat (embedding floatvector_l2_ops) WITH (ivf_nlist = {self._n_list})"
            )
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        print("done!")


    def set_query_arguments(self, n_probe):
        self._n_probe = n_probe
        self._cur.execute("SET enable_seqscan TO off; set ivf_probes = %d" % n_probe)

    def query(self, v, n):
        v_str = "[" + ", ".join(f"{x:.8f}" for x in v) + "]"
        self._cur.execute(self._query, (v_str, n))
        res = [id for id, in self._cur.fetchall()]
        return res

    def get_memory_usage(self):
        if self._cur is None:
            return 0
        self._cur.execute("SELECT pg_relation_size('items_embedding_idx')")
        return self._cur.fetchone()[0] / 1024

    def get_table_usage(self):
        self._cur.execute(f"SELECT pg_relation_size('{self.tablename}')")
        return self._cur.fetchone()[0] / 1024


    def __str__(self):
        return "openGauss-ivfflat(n_list=%d, n_probe=%d)" % (self._n_list, self._n_probe)


class OpenGaussIvfPQVector(OpenGaussIvfflatVector):
    def __init__(self, metric, nlist, m=32, nbits=8):
        super().__init__(metric, nlist)
        self._m = m  # 默认值
        self._nbits = nbits  # 默认值
        self._n_probe = None
        self._n_factor = None


    def set_query_arguments(self, n_probe, n_factor):
        self._n_probe = n_probe
        self._n_factor = n_factor
        self._cur.execute(
            f"SET enable_seqscan TO off; set ivf_probes = {n_probe}; set ivfpq_refine_k_factor={n_factor};")

    def create_index(self):
        """Override create_index for IVF-PQ"""
        self._cur.execute(f"DROP INDEX IF EXISTS {self.tablename}_embedding_idx;")
        print("creating pq index...")
        if self._metric == "angular":
            self._cur.execute(
                f"CREATE INDEX ON {self.tablename} USING ivfpq (embedding floatvector_cosine_ops) WITH (ivf_nlist = {self._n_list}, num_subquantizers = {self._m}, nbits = {self._nbits}, parallel_workers = {self.parallel_workers})"
            )
        elif self._metric == "euclidean":
            self._cur.execute(
                f"CREATE INDEX ON {self.tablename} USING ivfpq (embedding floatvector_l2_ops) WITH (ivf_nlist = {self._n_list}, num_subquantizers = {self._m}, nbits = {self._nbits}, parallel_workers = {self.parallel_workers})"
            )
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        print("done!")

    def __str__(self):
        return "opengauss-ivfpq(n_list=%d, n_probe=%d, ivfpq_refine_k_factor=%d, parallel_workers=%d, num_subquantizers=%d, nbits=%d)" % (self._n_list, self._n_probe, self._n_factor, self.parallel_workers, self._m, self._nbits)


class OpenGaussHnswVector(OpenGaussIvfflatVector):
    def __init__(self, metric, m, efConstruction):
        super().__init__(metric, 0)
        self._m = m
        self._ef_construction = efConstruction
        self._cur = None
        self.parallel_workers = 8  # 默认值
        self.tablename = "items"
        self._ef_search = None
        self.conn = None
        self.conn = get_conn()
        self._cur = self.conn.cursor()

        if metric == "angular":
            self._query = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"
        elif metric == "euclidean":
            self._query = "SELECT id FROM items ORDER BY embedding <-> %s LIMIT %s"
        else:
            raise RuntimeError(f"unknown metric {metric}")


    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search
        self._cur.execute(
            "SET enable_seqscan TO off; set hnsw_ef_search = %d" % ef_search)

    def create_index(self):
        """Override create_index for HNSW"""
        self._cur.execute(f"DROP INDEX IF EXISTS {self.tablename}_embedding_idx;")
        print("creating index...")
        if self._metric == "angular":
            self._cur.execute(
                f"CREATE INDEX ON {self.tablename} USING hnsw (embedding floatvector_cosine_ops) WITH (m = {self._m}, ef_construction = {self._ef_construction}, parallel_workers={self.parallel_workers})"
            )
        elif self._metric == "euclidean":
            self._cur.execute(
                f"CREATE INDEX ON {self.tablename} USING hnsw (embedding floatvector_l2_ops) WITH (m = {self._m}, ef_construction = {self._ef_construction}, parallel_workers={self.parallel_workers})"
            )
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        print("done!")


    def __str__(self):
        return f"opengauss-hnsw(m={self._m}, ef_construction={self._ef_construction}, hnsw_ef_search={self._ef_search})"


class OpenGaussDiskannVector(OpenGaussIvfflatVector):
    def __init__(self, metric):
        super().__init__(metric, 0)
        self._ef_search = None

    def create_index(self):
        self._cur.execute(f"DROP INDEX IF EXISTS {self.tablename}_embedding_idx;")
        print("creating index...")
        if self._metric == "angular":
            self._cur.execute(
                f"CREATE INDEX ON {self.tablename} USING diskann (embedding floatvector_cosine_ops) WITH (parallel_workers = {self.parallel_workers}, enable_quantization=off);")
        elif self._metric == "euclidean":
            self._cur.execute(
                f"CREATE INDEX ON {self.tablename} USING diskann (embedding floatvector_l2_ops) WITH (parallel_workers = {self.parallel_workers}, enable_quantization=off);")
        else:
            raise RuntimeError(f"unknown metric {self._metric}")

    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search
        self._cur.execute(
            "SET enable_seqscan TO off; set diskann_search_list_size = %d; set diskann_query_with_pq = off;" % ef_search)

    def __str__(self):
        return f"opengauss-diskann(diskann_search_list_size={self._ef_search})"
