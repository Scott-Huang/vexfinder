import pymysql
import io
from ..base.module import BaseANN


def get_conn():
    conn = pymysql.connect(host="127.0.0.1", user="root", passwd="hyjCG2JlVzJgHQKs3lVy", db="oceanbase", port=2881)
    conn.autocommit = True
    return conn


class OceanBaseHnswVector(BaseANN):
    def __init__(self, metric, m, efConstruction):
        super().__init__()
        self._m = m
        self._ef_construction = efConstruction
        self.parallel_workers = 8
        self._metric = metric
        self._cur = None
        self.tablename = "items"
        self._ef_search = None
        self.conn = None
        self.conn = get_conn()
        self._cur = self.conn.cursor()

        
        if metric == "angular":
            self._query = f"SELECT /*+ query_timeout(60000000) */ *  id FROM {self.tablename} ORDER BY embedding <=> %s LIMIT %s"
        elif metric == "euclidean":
            self._query = f"SELECT /*+ query_timeout(60000000) */ *  id FROM {self.tablename} ORDER BY embedding <-> %s LIMIT %s"
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
        # 创建表 - 修改为OceanBase的语法

        self._cur.execute(f"CREATE TABLE IF NOT EXISTS {self.tablename} (id int PRIMARY KEY, embedding VECTOR({train_data.shape[1]}))")

        total_rows = train_data.shape[0]        
        print(f"Start copying {total_rows} rows...")
        
        # 批量插入的大小
        batch_size = 10000
        
        for start_idx in range(0, total_rows, batch_size):
            # 计算当前批次的结束索引
            end_idx = min(start_idx + batch_size, total_rows)
            
            # 构建批量插入的值字符串
            value_strings = []
            for i in range(start_idx, end_idx):
                embedding = train_data[i]
                embedding_str = "[" + ", ".join(f"{x:.8f}".rstrip("0").rstrip(".") for x in embedding) + "]"
                value_strings.append(f"({i}, '{embedding_str}')")
            
            # 合并所有值为一个批量插入语句
            values_clause = ", ".join(value_strings)
            insert_sql = f"INSERT INTO {self.tablename} (id, embedding) VALUES {values_clause}"
            
            # 执行批量插入
            self._cur.execute(insert_sql)
            
            # 打印进度
            if end_idx % (total_rows // 10) == 0 or end_idx == total_rows:
                print(f"Processed {end_idx}/{total_rows} rows ({(end_idx / total_rows * 100):.2f}%)")

    def create_index(self):
        """Create HNSW index"""
        try:
            # 尝试删除索引，如果存在的话
            self._cur.execute(f"DROP INDEX {self.tablename}_embedding_idx ON {self.tablename}")
        except:
            # 如果索引不存在，忽略错误
            pass
            
        print("creating index...")
        self._cur.execute("ALTER SYSTEM SET ob_vector_memory_limit_percentage = 30")
        if self._metric == "angular":
            self._cur.execute(
                f"CREATE /*+ PARALLEL({self.parallel_workers}) */ VECTOR INDEX {self.tablename}_embedding_idx ON {self.tablename} (embedding) with (distance=cosine, type=hnsw, lib=vsag, m={self._m}, ef_construction={self._ef_construction})"
            )
        elif self._metric == "euclidean":
            self._cur.execute(
                f"CREATE /*+ PARALLEL({self.parallel_workers}) */ VECTOR INDEX {self.tablename}_embedding_idx ON {self.tablename} (embedding) with (distance=l2, type=hnsw, lib=vsag, m={self._m}, ef_construction={self._ef_construction})"
            )
        else:
            raise RuntimeError(f"unknown metric {self._metric}")
        print("done!")


    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search
        self._cur.execute(f"SET ob_hnsw_ef_search = {ef_search}")

    def query(self, v, n):
        v_str = "[" + ", ".join(f"{x:.8f}" for x in v) + "]"
        self._cur.execute(
            f"SELECT id FROM {self.tablename} ORDER BY l2_distance(embedding, '{v_str}') APPROXIMATE LIMIT {n}"
        )
        res = [idx[0] for idx in self._cur.fetchall()]
        return res

    def get_memory_usage(self):
        return 1
        # if self._cur is None:
        #     return 0
        # self._cur.execute(f"SELECT table_id FROM __all_virtual_table WHERE table_name = '{self.tablename}';")
        # table_id = self._cur.fetchone()[0]
        # self._cur.execute(f"SELECT * FROM __all_virtual_storage_stat WHERE table_id = {table_id};")
        # return self._cur.fetchone()[0] / 1024

    def get_table_usage(self):
        return 1
        # if self._cur is None:
        #     return 0
        # self._cur.execute(f"SELECT table_id FROM __all_virtual_table WHERE data_table_id = '{self.tablename}';")
        # table_id = self._cur.fetchone()[0]
        # self._cur.execute(f"SELECT * FROM __all_virtual_storage_stat WHERE table_id = {table_id};")
        # return self._cur.fetchone()[0] / 1024


    def __str__(self):
        return f"oceanbase-hnsw(m={self._m}, ef_construction={self._ef_construction}, hnsw_ef_search={self._ef_search})"



