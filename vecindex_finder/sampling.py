import core.engine as db_engine
from core.config import Config
from core.logging import logger
import queue
import threading
import tqdm
import json

class Sampling:
    """采样器，负责管理采样"""
    
    def __init__(self, config: Config, db_engine: db_engine.DatabaseEngine):
        self.sampling_config = config.sampling_config
        self.parallel_workers = config.parallel_workers
        self.table_info_config = config.table_info_config
        self.query_config = config.query_config
        self.performance_config = config.performance_config
        self.index_config = config.index_config
        self.db_engine = db_engine

        self.table_name = self.table_info_config['table_name']
        self.vector_column_name = self.table_info_config['vector_column_name']
        self.dimension = self.table_info_config['dimension']
        self.metric = self.table_info_config['metric']

        self.sample_table_name = self.table_info_config['sample_table_name']
        self.query_table_name = self.table_info_config['query_table_name']

        self.default_ratio = self.sampling_config['default_ratio']
        self.min_sample_count = self.sampling_config['min_sample_count']
        self.max_sample_count = self.sampling_config['max_sample_count']

        self.query_get_type = self.query_config['query_get_type']
        self.query_data_path = self.query_config['query_data_path']
        self.query_count = self.query_config['query_count']

        self.limit = self.performance_config['limit']

    
    def sampling_data(self) -> str:
        """
        从指定表采样数据并存储到新表中        
        Returns:
            采样表名
        """
        connection = self.db_engine.get_connection()
        cursor = connection.cursor()
        # 检查表是否存在
        cursor.execute(f"SELECT 1 FROM pg_tables WHERE tablename = '{self.table_name}'")
        if not cursor.fetchone():
            raise ValueError(f"表 {self.table_name} 不存在")
        
        # 检查配置文件中指定的向量列是否存在
        cursor.execute(f"""
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = '{self.table_name}' 
            AND column_name = '{self.vector_column_name}'
        """)
        result = cursor.fetchone()
        if not result:
            raise ValueError(f"表 {self.table_name} 中未找到配置的向量列 {self.vector_column_name}")
        
        # 获取表大小
        cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        total_count = cursor.fetchone()[0]
        
        # 根据配置文件中的默认采样比例计算采样数量
        sample_count = int(total_count * self.default_ratio)
        # 根据配置文件中的最小和最大采样数量调整采样数量
        sample_count = int(total_count * self.default_ratio)
        if sample_count < self.min_sample_count and total_count >= self.min_sample_count:
            sample_count = self.min_sample_count
            logger.info(f"采样数量小于最小采样数量，调整为: {self.min_sample_count}")
        if sample_count > self.max_sample_count:
            sample_count = self.max_sample_count
            logger.info(f"采样数量大于最大采样数量，调整为: {self.max_sample_count}")
        
        # 确保采样数量不超过总数量
        sample_count = min(sample_count, total_count)
        
        logger.info(f"从表 {self.table_name} 采样 {sample_count}/{total_count} 条数据")

        # 创建采样表（不创建索引）
        cursor.execute(f"DROP TABLE IF EXISTS {self.sample_table_name}")
        cursor.execute(f"""
            CREATE TABLE {self.sample_table_name} (
                id SERIAL PRIMARY KEY,
                {self.vector_column_name} floatvector({self.dimension})
            )
        """)
        
        logger.info(f"创建采样表 {self.sample_table_name}")
        
        # 采样数据并插入到新表
        cursor.execute(f"""
            INSERT INTO {self.sample_table_name} ({self.vector_column_name})
            SELECT {self.vector_column_name} FROM {self.table_name} 
            ORDER BY RANDOM() LIMIT {sample_count}
        """)
        
        # 获取实际采样数量
        cursor.execute(f"SELECT COUNT(*) FROM {self.sample_table_name}")
        actual_count = cursor.fetchone()[0]
        
        logger.info(f"采样完成，共插入 {actual_count} 条向量数据到表 {self.sample_table_name}")

        connection.close()
        return self.sample_table_name

    def sampling_query_data(self) -> str:
        """
        采样查询数据并存储到新表中
        Returns:
            查询表名
        """
        connection = self.db_engine.get_connection()
        cursor = connection.cursor()
        # drop query table if exists
        cursor.execute(f"DROP TABLE IF EXISTS {self.query_table_name}")

        # 创建查询表（包含一个用于存储最近邻距离的distances列）
        cursor.execute(f"""
            CREATE TABLE {self.query_table_name} (
                id SERIAL PRIMARY KEY,
                {self.vector_column_name} floatvector({self.dimension}),
                distances text DEFAULT NULL
            )
        """)
        logger.info(f"创建查询表 {self.query_table_name}")

        if self.query_get_type == 'sample':
            # 采样数据并插入到新表
            query = f"""
                INSERT INTO {self.query_table_name} (id, {self.vector_column_name})
                SELECT id, {self.vector_column_name} FROM {self.table_name} 
                ORDER BY RANDOM() LIMIT {self.query_count}
            """
            cursor.execute(query)
        elif self.query_get_type == 'json':
            # 从json文件中读取查询数据
            with open(self.query_data_path, 'r') as f:
                query_data = json.load(f)
            for i, query in enumerate(query_data):
                # 修复SQL注入风险，使用参数化查询
                cursor.execute(
                    f"INSERT INTO {self.query_table_name} (id, {self.vector_column_name}) VALUES (%s, %s)",
                    (i, query)
                )
        # 获取实际采样数量
        cursor.execute(f"SELECT COUNT(*) FROM {self.query_table_name}")
        actual_count = cursor.fetchone()[0]
        logger.info(f"查询数据采样完成，共插入 {actual_count} 条向量数据到表 {self.query_table_name}")
        connection.close()
        return self.query_table_name
    
    def compute_sample_query_distance(self) -> None:
        """
        计算测试表中每个向量在训练表中的最近距离，并将结果存储在测试表中
        使用线程池消费者模式，每个线程从共享队列中获取任务
        """
        connection = self.db_engine.get_connection()
        cursor = connection.cursor()
        
        # 检查表是否存在
        for table in [self.sample_table_name, self.query_table_name]:
            cursor.execute(f"SELECT 1 FROM pg_tables WHERE tablename = '{table}'")
            if not cursor.fetchone():
                raise ValueError(f"表 {table} 不存在")

        # 确定距离操作符
        if self.metric == "cosine":
            distance_operator = "<=>"  # 余弦距离
        elif self.metric == "l2":
            distance_operator = "<->"  # 欧氏距离
        elif self.metric == "ip":
            distance_operator = "<#>"  # 内积距离
        else:
            raise ValueError(f"不支持的距离度量类型: {self.metric}")
        
        # 获取测试表中的所有ID和向量
        cursor.execute(f"SELECT id, {self.vector_column_name} FROM {self.query_table_name}")
        test_data = cursor.fetchall()
        total_vectors = len(test_data)
        
        logger.info(f"开始计算 {total_vectors} 个查询向量的 {self.limit} 个最近邻")
        
        # 计算并行度，确保parallel_workers不为None
        if self.parallel_workers is None:
            self.parallel_workers = 8
            logger.info(f"parallel_workers未设置，使用默认值: {self.parallel_workers}")
        
        # 创建任务队列
        task_queue = queue.Queue()
        
        # 将所有任务放入队列
        for test_id, test_vector in test_data:
            task_queue.put((test_id, test_vector))
        
        # 创建结果列表和锁
        results = []
        results_lock = threading.Lock()
        
        # 创建进度计数器和锁
        progress_counter = 0
        counter_lock = threading.Lock()
        
        # 创建进度条
        progress_bar = tqdm.tqdm(total=total_vectors, desc="计算最近邻")
        
        # 工作线程函数
        def worker():
            # 创建单独的数据库连接
            conn = self.db_engine.get_connection()
            cur = conn.cursor()
            
            while True:
                try:
                    # 从队列中获取任务，如果队列为空则退出
                    try:
                        test_id, vector_str = task_queue.get(block=False)
                    except queue.Empty:
                        break
                    
                    # 执行查询
                    cur.execute(f"""
                        SELECT id, {self.vector_column_name} {distance_operator} %s AS distance 
                        FROM {self.sample_table_name} 
                        ORDER BY distance ASC 
                        LIMIT {self.limit}
                    """, (vector_str,))
                    
                    # 获取结果 - 只保存距离数组
                    distances = [float(row[1]) for row in cur.fetchall()]
                    
                    # 将结果添加到结果列表
                    with results_lock:
                        results.append((test_id, distances))
                    
                    # 更新进度计数器和进度条
                    with counter_lock:
                        nonlocal progress_counter
                        progress_counter += 1
                        progress_bar.update(1)
                    
                    # 标记任务完成
                    task_queue.task_done()
                    
                except Exception as e:
                    logger.error(f"处理查询时出错: {e}")
                    # 标记任务完成，即使出错
                    task_queue.task_done()
            
            # 关闭数据库连接
            cur.close()
            conn.close()
        
        # 创建并启动工作线程
        threads = []
        for _ in range(self.parallel_workers):
            thread = threading.Thread(target=worker)
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        # 等待所有任务完成
        for thread in threads:
            thread.join()
        
        # 关闭进度条
        progress_bar.close()
        
        # 更新数据库
        logger.info("更新数据库中的距离信息...")
        for test_id, distances in results:
            distances_json = json.dumps(distances)
            cursor.execute(
                f"UPDATE {self.query_table_name} SET distances = %s WHERE id = %s",
                (distances_json, test_id)
            )
        
        logger.info("最近邻计算完成")
        connection.close()