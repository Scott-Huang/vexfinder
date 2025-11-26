from core.engine import db_engine, DatabaseEngine
from core.config import Config, config
from core.logging import logger
from typing import Optional
import queue
import threading
import tqdm
import json
from core.my_types import QueryData

class Sampling:
    def __init__(self, config_obj: Optional[Config] = None, db_engine_obj: Optional[DatabaseEngine] = None):
        self.config = config_obj or config
        self.db_engine = db_engine_obj or db_engine

    def sampling_data(self) -> str:
        connection = self.db_engine.get_connection()
        cursor = connection.cursor()
        cursor.execute(f"SELECT 1 FROM pg_tables WHERE tablename = '{self.config.table_info.table_name}'")
        if not cursor.fetchone():
            raise ValueError(f"表 {self.config.table_info.table_name} 不存在")

        cursor.execute(f"""
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = '{self.config.table_info.table_name}' 
            AND column_name = '{self.config.table_info.vector_column_name}'
        """)
        result = cursor.fetchone()
        if not result:
            raise ValueError(f"表 {self.config.table_info.table_name} 中未找到配置的向量列 {self.config.table_info.vector_column_name}")

        cursor.execute(f"SELECT COUNT(1) FROM {self.config.table_info.table_name}")
        total_count = cursor.fetchone()[0]
        self.config.table_info.original_table_count = total_count

        sample_count = int(total_count * self.config.sampling.default_ratio)
        if sample_count < self.config.sampling.min_sample_count and total_count >= self.config.sampling.min_sample_count:
            sample_count = self.config.sampling.min_sample_count
            logger.info(f"采样数量小于最小采样数量，调整为: {self.config.sampling.min_sample_count}")
        if sample_count > self.config.sampling.max_sample_count:
            sample_count = self.config.sampling.max_sample_count
            logger.info(f"采样数量大于最大采样数量，调整为: {self.config.sampling.max_sample_count}")

        sample_count = min(sample_count, total_count)
        logger.info(f"从表 {self.config.table_info.table_name} 采样 {sample_count}/{total_count} 条数据")

        cursor.execute(f"DROP TABLE IF EXISTS {self.config.table_info.sample_table_name}")
        cursor.execute(f"""
            CREATE TABLE {self.config.table_info.sample_table_name} (
                id SERIAL PRIMARY KEY,
                {self.config.table_info.vector_column_name} floatvector({self.config.table_info.dimension})
            )"""
        )

        logger.info(f"创建采样表 {self.config.table_info.sample_table_name}")

        cursor.execute(f"""
            INSERT INTO {self.config.table_info.sample_table_name} ({self.config.table_info.vector_column_name})
            SELECT {self.config.table_info.vector_column_name} FROM {self.config.table_info.table_name} 
            ORDER BY RANDOM() LIMIT {sample_count}
        """)
        cursor.execute(f"ANALYZE {self.config.table_info.sample_table_name}")

        cursor.execute(f"SELECT COUNT(*) FROM {self.config.table_info.sample_table_name}")
        actual_count = cursor.fetchone()[0]
        self.config.table_info.sample_table_count = actual_count
        logger.info(f"采样完成，共插入 {actual_count} 条向量数据到表 {self.config.table_info.sample_table_name}")
        connection.close()
        return actual_count

    def sampling_query_data(self) -> str:
        connection = db_engine.get_connection()
        cursor = connection.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {self.config.table_info.query_table_name}")

        cursor.execute(f"""
            CREATE TABLE {self.config.table_info.query_table_name} (
                id SERIAL PRIMARY KEY,
                {self.config.table_info.vector_column_name} floatvector({self.config.table_info.dimension}),
                distances text DEFAULT NULL
            )"""
        )
        logger.info(f"创建查询表 {self.config.table_info.query_table_name}")

        if self.config.query.query_get_type == 'sample':
            query = f"""
                INSERT INTO {self.config.table_info.query_table_name} ({self.config.table_info.vector_column_name})
                SELECT {self.config.table_info.vector_column_name}
                FROM {self.config.table_info.table_name}
                ORDER BY RANDOM() LIMIT {self.config.query.query_count}
            """
            cursor.execute(query)
        elif self.config.query.query_get_type == 'json':
            with open(self.config.query.query_data_path, 'r') as f:
                query_data = json.load(f)
            for i, query in enumerate(query_data):
                cursor.execute(
                    f"INSERT INTO {self.config.table_info.query_table_name} (id, {self.config.table_info.vector_column_name}) VALUES (%s, %s)",
                    (i, query)
                )

        cursor.execute(f"ANALYZE {self.config.table_info.query_table_name}")
        cursor.execute(f"SELECT COUNT(*) FROM {self.config.table_info.query_table_name}")
        actual_count = cursor.fetchone()[0]
        self.config.table_info.query_table_count = actual_count
        logger.info(f"查询数据采样完成，共插入 {actual_count} 条向量数据到表 {self.config.table_info.query_table_name}")
        connection.close()
        return actual_count

    def get_all_query_data(self) -> list[QueryData]:
        connection = self.db_engine.get_connection()
        cursor = connection.cursor()
        cursor.execute(f"SELECT id, {self.config.table_info.vector_column_name}, distances FROM {self.config.table_info.query_table_name}")
        query_data = cursor.fetchall()
        connection.close()
        return [QueryData(id=row[0], vectors=json.loads(row[1]), distances=json.loads(row[2])) for row in query_data]

    def get_table_count(self, table_name: str) -> int:
        connection = self.db_engine.get_connection()
        cursor = connection.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        actual_count = cursor.fetchone()[0]
        connection.close()
        return actual_count

    def get_original_table_count(self) -> int:
        return self.get_table_count(self.config.table_info.table_name)

    def get_sample_table_count(self) -> int:
        return self.get_table_count(self.config.table_info.sample_table_name)

    def get_query_table_count(self) -> int:
        return self.get_table_count(self.config.table_info.query_table_name)

    def get_query_table_count(self) -> int:
        connection = self.db_engine.get_connection()
        cursor = connection.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.config.table_info.query_table_name}")
        actual_count = cursor.fetchone()[0]
        self.config.table_info.query_table_count = actual_count
        logger.info(f"共有 {actual_count} 条查询语句")
        connection.close()
        return actual_count

    def compute_sample_query_distance(self) -> None:
        connection = self.db_engine.get_connection()
        cursor = connection.cursor()

        for table in [self.config.table_info.sample_table_name, self.config.table_info.query_table_name]:
            cursor.execute(f"SELECT 1 FROM pg_tables WHERE tablename = '{table}'")
            if not cursor.fetchone():
                raise ValueError(f"表 {table} 不存在")

        if self.config.table_info.metric == "cosine":
            distance_operator = "<=>"
        elif self.config.table_info.metric == "l2":
            distance_operator = "<->"
        elif self.config.table_info.metric == "ip":
            distance_operator = "<#>"
        else:
            raise ValueError(f"不支持的距离度量类型: {self.config.table_info.metric}")

        cursor.execute(f"SELECT id, {self.config.table_info.vector_column_name} FROM {self.config.table_info.query_table_name}")
        test_data = cursor.fetchall()
        total_vectors = len(test_data)

        logger.info(f"开始计算 {total_vectors} 个查询向量的 {self.config.performance.limit} 个最近邻")
        if self.config.parallel_workers is None:
            self.config.parallel_workers = 8
            logger.info(f"parallel_workers未设置，使用默认值: {self.config.parallel_workers}")

        task_queue = queue.Queue()
        for test_id, test_vector in test_data:
            task_queue.put((test_id, test_vector))

        results = []
        results_lock = threading.Lock()

        progress_counter = 0
        counter_lock = threading.Lock()

        progress_bar = tqdm.tqdm(total=total_vectors, desc="计算最近邻")

        def worker():
            conn = self.db_engine.get_connection()
            cur = conn.cursor()
            
            while True:
                try:
                    try:
                        test_id, vector_str = task_queue.get(block=False)
                    except queue.Empty:
                        break
                    
                    # 执行查询
                    cur.execute(f"""
                        SELECT id, {self.config.table_info.vector_column_name} {distance_operator} %s AS distance
                        FROM {self.config.table_info.sample_table_name}
                        ORDER BY distance ASC
                        LIMIT {self.config.performance.limit}
                    """, (vector_str,))

                    distances = [float(row[1]) for row in cur.fetchall()]

                    with results_lock:
                        results.append((test_id, distances))

                    with counter_lock:
                        nonlocal progress_counter
                        progress_counter += 1
                        progress_bar.update(1)

                    task_queue.task_done()

                except Exception as e:
                    logger.error(f"处理查询时出错: {e}")
                    task_queue.task_done()

            cur.close()
            conn.close()

        threads = []
        for _ in range(self.config.parallel_workers):
            thread = threading.Thread(target=worker)
            thread.daemon = True
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        progress_bar.close()

        logger.info("更新数据库中的距离信息...")
        for test_id, distances in results:
            distances_json = json.dumps(distances)
            cursor.execute(
                f"UPDATE {self.config.table_info.query_table_name} SET distances = %s WHERE id = %s",
                (distances_json, test_id)
            )

        logger.info("最近邻计算完成")
        connection.close()
