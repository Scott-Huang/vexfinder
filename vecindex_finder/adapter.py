#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import logging
import json
import tqdm
from typing import Dict, List, Any, Tuple, Optional
import threading
import queue
from core.engine import db_engine
from core.config import Config

logger = logging.getLogger(__name__)

class OpenGaussAdapter:
    """OpenGauss数据库适配器"""
    
    def __init__(self):
        """初始化OpenGauss适配器"""
        self.connection = None
        self.cursor = None
        self.metric_type = None
                    
        # 获取配置管理器
        self.config_manager = db_engine.get_config_manager()
        
        # 从配置管理器获取配置
        self.table_name, self.vector_column_name, self.dimension, self.metric = self.config_manager.get_table_info()
        self.config = self.config_manager.config
        
        # 获取其他配置
        index_config = self.config_manager.get_index_config()
        self._find_index_type = index_config['find_index_type']
        self._find_index_auto = index_config['auto']
        
        performance_config = self.config_manager.get_performance_config()
        self._limit = performance_config['limit']
        self._min_recall = performance_config['min_recall']
        self._max_recall = performance_config['max_recall']
        
        query_config = self.config_manager.get_query_config()
        self._query_get_type = query_config['query_get_type']
        self._query_data_path = query_config['query_data_path']
        self._query_count = query_config['query_count']
        self._query_table_name = query_config['query_table_name']
        
        sampling_config = self.config_manager.get_sampling_config()
        self._sample_table_name = sampling_config['sample_table_name']
        
        self.parallel_workers = self.config_manager.parallel_workers
    
    def load_and_check_config(self):
        """
        加载配置文件并检查必要配置项
        
        注意：此方法已被废弃，配置管理已移至config.py中的OpenGaussConfig类
        """
        logger.warning("load_and_check_config方法已被废弃，配置管理已移至config.py中的OpenGaussConfig类")
        pass

    def get_connection(self):
        """
        获取数据库连接
        """
        return db_engine.get_connection()


    def connect(self):
        """
        连接到OpenGauss服务器
        
        Returns:
            self: 返回自身实例以支持链式调用
        """
        # 如果没有提供配置，则使用配置文件中的配置
        if self.config is None:
            raise ValueError("未提供连接配置，且配置文件中没有连接配置")
        
        logger.info(f"连接到OpenGauss服务器: {self.config['connection']['host']}:{self.config['connection']['port']}")
        
        try:
            self.connection = self.get_connection()
            self.cursor = self.connection.cursor()
            logger.info("OpenGauss连接成功")
            return self
        except Exception as e:
            logger.error(f"OpenGauss连接失败: {e}")
            raise ConnectionError(f"无法连接到OpenGauss服务器: {e}")
    
    def sample_data(self) -> str:
        """
        从指定表采样数据并存储到新表中        
        Returns:
            采样表名
        """
        # 检查表是否存在
        self.cursor.execute(f"SELECT 1 FROM pg_tables WHERE tablename = '{self.table_name}'")
        if not self.cursor.fetchone():
            raise ValueError(f"表 {self.table_name} 不存在")
        
        # 检查配置文件中指定的向量列是否存在
        self.cursor.execute(f"""
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = '{self.table_name}' 
            AND column_name = '{self.vector_column_name}'
        """)
        result = self.cursor.fetchone()
        if not result:
            raise ValueError(f"表 {self.table_name} 中未找到配置的向量列 {self.vector_column_name}")
        
        # 获取表大小
        self.cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        total_count = self.cursor.fetchone()[0]
        
        # 根据配置文件中的默认采样比例计算采样数量
        sample_ratio = self.config['sampling']['default_ratio']
        
        # 根据配置文件中的最小和最大采样数量调整采样数量
        sample_count = int(total_count * sample_ratio)
        min_sample_count = self.config['sampling']['min_sample_count']
        max_sample_count = self.config['sampling']['max_sample_count']
        if sample_count < min_sample_count and total_count >= min_sample_count:
            sample_count = min_sample_count
            logger.info(f"采样数量小于最小采样数量，调整为: {min_sample_count}")
        if sample_count > max_sample_count:
            sample_count = max_sample_count
            logger.info(f"采样数量大于最大采样数量，调整为: {max_sample_count}")
        
        # 确保采样数量不超过总数量
        sample_count = min(sample_count, total_count)
        
        logger.info(f"从表 {self.table_name} 采样 {sample_count}/{total_count} 条数据")
        
        # 创建采样表（不创建索引）
        self.cursor.execute(f"DROP TABLE IF EXISTS {self._sample_table_name}")
        self.cursor.execute(f"""
            CREATE TABLE {self._sample_table_name} (
                id SERIAL PRIMARY KEY,
                {self.vector_column_name} floatvector({self.dimension})
            )
        """)
        
        logger.info(f"创建采样表 {self._sample_table_name}")
        
        # 采样数据并插入到新表
        self.cursor.execute(f"""
            INSERT INTO {self._sample_table_name} ({self.vector_column_name})
            SELECT {self.vector_column_name} FROM {self.table_name} 
            ORDER BY RANDOM() LIMIT {sample_count}
        """)
        
        # 获取实际采样数量
        self.cursor.execute(f"SELECT COUNT(*) FROM {self._sample_table_name}")
        actual_count = self.cursor.fetchone()[0]
        
        logger.info(f"采样完成，共插入 {actual_count} 条向量数据到表 {self._sample_table_name}")
        return self._sample_table_name
    
    def sample_query_data(self) -> str:
        """
        采样查询数据并存储到新表中
        Returns:
            查询表名
        """
        # drop query table if exists
        self.cursor.execute(f"DROP TABLE IF EXISTS {self._query_table_name}")

        # 创建查询表（包含一个用于存储最近邻距离的distances列）
        self.cursor.execute(f"""
            CREATE TABLE {self._query_table_name} (
                id SERIAL PRIMARY KEY,
                {self.vector_column_name} floatvector({self.dimension}),
                distances text DEFAULT NULL
            )
        """)
        logger.info(f"创建查询表 {self._query_table_name}")

        if self._query_get_type == 'sample':
            # 采样数据并插入到新表
            query = f"""
                INSERT INTO {self._query_table_name} (id, {self.vector_column_name})
                SELECT id, {self.vector_column_name} FROM {self.table_name} 
                ORDER BY RANDOM() LIMIT {self._query_count}
            """
            self.cursor.execute(query)
        elif self._query_get_type == 'json':
            # 从json文件中读取查询数据
            with open(self._query_data_path, 'r') as f:
                query_data = json.load(f)
            for i, query in enumerate(query_data):
                # 修复SQL注入风险，使用参数化查询
                self.cursor.execute(
                    f"INSERT INTO {self._query_table_name} (id, {self.vector_column_name}) VALUES (%s, %s)",
                    (i, query)
                )
        # 获取实际采样数量
        self.cursor.execute(f"SELECT COUNT(*) FROM {self._query_table_name}")
        actual_count = self.cursor.fetchone()[0]
        logger.info(f"查询数据采样完成，共插入 {actual_count} 条向量数据到表 {self._query_table_name}")
        return self._query_table_name
    
    def compute_query_nearest_distance(self) -> None:
        """
        计算测试表中每个向量在训练表中的最近距离，并将结果存储在测试表中
        使用线程池消费者模式，每个线程从共享队列中获取任务
        """
        
        # 检查表是否存在
        for table in [self._sample_table_name, self._query_table_name]:
            self.cursor.execute(f"SELECT 1 FROM pg_tables WHERE tablename = '{table}'")
            if not self.cursor.fetchone():
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
        self.cursor.execute(f"SELECT id, {self.vector_column_name} FROM {self._query_table_name}")
        test_data = self.cursor.fetchall()
        total_vectors = len(test_data)
        
        # 如果_limit未设置，默认为100
        if self._limit is None:
            self._limit = 100
            logger.info(f"_limit未设置，使用默认值: {self._limit}")
        
        logger.info(f"开始计算 {total_vectors} 个查询向量的 {self._limit} 个最近邻")
        
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
            conn = self.get_connection()
            cur = conn.cursor()
            
            while True:
                try:
                    # 从队列中获取任务，如果队列为空则退出
                    try:
                        test_id, test_vector = task_queue.get(block=False)
                    except queue.Empty:
                        break
                    
                    # 将向量格式化为OpenGauss接受的格式
                    vector_str = str(test_vector)
                    
                    # 执行查询
                    cur.execute(f"""
                        SELECT id, {self.vector_column_name} {distance_operator} %s AS distance 
                        FROM {self._sample_table_name} 
                        ORDER BY distance ASC 
                        LIMIT {self._limit}
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
            self.cursor.execute(
                f"UPDATE {self._query_table_name} SET distances = %s WHERE id = %s",
                (distances_json, test_id)
            )
        
        logger.info("最近邻计算完成")
    
    def get_groundtruth_results(self, test_table: str) -> List[List[int]]:
        """
        获取真实结果
        
        Args:
            test_table: 真实结果表名
            
        Returns:
            真实结果列表，每个查询的结果是一个ID列表
        """
        # 为获取真实结果创建新的连接
        connection = self.get_connection()
        cursor = connection.cursor()
        
        try:
            # 检查表是否存在
            cursor.execute(f"SELECT 1 FROM pg_tables WHERE tablename = '{test_table}'")
            if not cursor.fetchone():
                raise ValueError(f"表 {test_table} 不存在")
            
            # 获取表中的所有结果
            cursor.execute(f"SELECT query_id, result_ids FROM {test_table} ORDER BY query_id")
            results = cursor.fetchall()
            
            if not results:
                raise ValueError(f"表 {test_table} 中没有数据")
            
            # 解析结果
            groundtruth_results = []
            for _, result_ids in results:
                # 将字符串形式的ID列表转换为实际的ID列表
                if isinstance(result_ids, str):
                    # 如果是JSON字符串，解析它
                    try:
                        import json
                        ids = json.loads(result_ids)
                    except json.JSONDecodeError:
                        # 如果不是JSON，尝试其他格式
                        ids = [int(id.strip()) for id in result_ids.strip('{}').split(',') if id.strip()]
                else:
                    # 如果已经是列表形式，直接使用
                    ids = result_ids
                
                groundtruth_results.append(ids)
            
            return groundtruth_results
        finally:
            cursor.close()
            connection.close()
    
    def compute_exact_search(self, query_vectors: np.ndarray, index_vectors: np.ndarray, 
                           k: int, metric: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算精确搜索结果（暴力搜索）
        
        Args:
            query_vectors: 查询向量
            index_vectors: 索引向量
            k: 返回的最近邻数量
            metric: 距离度量类型 ('l2', 'ip', 'cosine')
            
        Returns:
            (indices, distances): 索引和距离的元组
        """
        logger.info(f"开始精确搜索，查询向量: {len(query_vectors)}，索引向量: {len(index_vectors)}，k={k}")
        
        # 初始化结果数组
        num_queries = len(query_vectors)
        indices = np.zeros((num_queries, k), dtype=np.int32)
        distances = np.zeros((num_queries, k), dtype=np.float32)
        
        # 根据不同的度量类型计算距离
        start_time = time.time()
        
        for i, query in enumerate(query_vectors):
            if i % 100 == 0:
                logger.debug(f"正在处理查询 {i}/{num_queries}")
                
            if metric == 'l2' or metric == 'euclidean':
                # 计算欧氏距离
                dist = np.sum((index_vectors - query) ** 2, axis=1)
                
                # 获取最近的k个点
                idx = np.argsort(dist)[:k]
                indices[i] = idx
                distances[i] = np.sqrt(dist[idx])  # 欧氏距离需要开方
                
            elif metric == 'ip' or metric == 'inner_product':
                # 计算内积，内积越大越相似，所以取负排序
                dist = -np.dot(index_vectors, query)
                
                # 获取内积最大的k个点
                idx = np.argsort(dist)[:k]
                indices[i] = idx
                distances[i] = dist[idx]
                
            elif metric == 'cosine' or metric == 'angular':
                # 计算余弦相似度
                norm_query = np.linalg.norm(query)
                norm_index = np.linalg.norm(index_vectors, axis=1)
                cosine_sim = np.dot(index_vectors, query) / (norm_index * norm_query)
                
                # 余弦相似度转换为距离，距离 = 1 - 相似度
                dist = 1 - cosine_sim
                
                # 获取距离最小的k个点
                idx = np.argsort(dist)[:k]
                indices[i] = idx
                distances[i] = dist[idx]
                
            else:
                raise ValueError(f"不支持的距离度量类型: {metric}")
                
        elapsed_time = time.time() - start_time
        logger.info(f"精确搜索完成，耗时: {elapsed_time:.2f}秒")
        
        return indices, distances
    
    def create_index(self, table_name: str, index_type: str, index_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        在指定表上创建向量索引
        
        Args:
            table_name: 表名
            index_type: 索引类型 ('ivfflat', 'ivfpq', 'hnsw')
            index_params: 索引参数，如果为None则使用配置文件中的默认参数
            
        Returns:
            包含索引信息的字典
        """
        # 为每个操作创建新的连接
        connection = self.get_connection()
        cursor = connection.cursor()
        
        try:
            # 检查表是否存在
            cursor.execute(f"SELECT 1 FROM pg_tables WHERE tablename = '{table_name}'")
            if not cursor.fetchone():
                raise ValueError(f"表 {table_name} 不存在")
            
            # 获取向量字段名
            cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}' AND data_type = 'floatvector'")
            result = cursor.fetchone()
            if not result:
                raise ValueError(f"表 {table_name} 中未找到向量字段")
            
            vector_column = result[0]
            
            # 获取维度信息
            
            # 检查是否已有索引
            cursor.execute(f"SELECT indexname FROM pg_indexes WHERE tablename = '{table_name}'")
            existing_indexes = cursor.fetchall()
            
            if existing_indexes:
                logger.warning(f"表 {table_name} 已有索引，将先删除")
                for index_name in [row[0] for row in existing_indexes]:
                    cursor.execute(f"DROP INDEX {index_name}")
            
            # 如果没有提供索引参数，则使用配置文件中的默认参数
            if index_params is None:
                index_params = {}
                
                if self.config and 'index_types' in self.config and index_type in self.config['index_types']:
                    index_config = self.config['index_types'][index_type]
                    
                    # 根据索引类型设置默认参数
                    if index_type == 'ivfflat':
                        # 使用配置中的第一个值作为默认值
                        if 'nlist_range' in index_config and index_config['nlist_range']:
                            index_params['nlist'] = index_config['nlist_range'][0]
                        else:
                            index_params['nlist'] = 100
                        
                        if 'nprobe_range' in index_config and index_config['nprobe_range']:
                            index_params['nprobe'] = index_config['nprobe_range'][0]
                        else:
                            index_params['nprobe'] = 16
                            
                    elif index_type == 'ivfpq':
                        if 'nlist_range' in index_config and index_config['nlist_range']:
                            index_params['nlist'] = index_config['nlist_range'][0]
                        else:
                            index_params['nlist'] = 100
                        
                        if 'm_range' in index_config and index_config['m_range']:
                            index_params['m'] = index_config['m_range'][0]
                        else:
                            index_params['m'] = 8
                        
                        if 'nbits' in index_config:
                            index_params['nbits'] = index_config['nbits']
                        else:
                            index_params['nbits'] = 8
                        
                        if 'nprobe_range' in index_config and index_config['nprobe_range']:
                            index_params['nprobe'] = index_config['nprobe_range'][0]
                        else:
                            index_params['nprobe'] = 16
                        
                        if 'ivfpq_refine_k_factor_range' in index_config and index_config['ivfpq_refine_k_factor_range']:
                            index_params['ivfpq_refine_k_factor'] = index_config['ivfpq_refine_k_factor_range'][0]
                        else:
                            index_params['ivfpq_refine_k_factor'] = 1
                            
                    elif index_type == 'hnsw':
                        if 'M_range' in index_config and index_config['M_range']:
                            index_params['M'] = index_config['M_range'][0]
                        else:
                            index_params['M'] = 16
                        
                        if 'efConstruction_range' in index_config and index_config['efConstruction_range']:
                            index_params['efConstruction'] = index_config['efConstruction_range'][0]
                        else:
                            index_params['efConstruction'] = 200
                        
                        if 'ef_range' in index_config and index_config['ef_range']:
                            index_params['ef'] = index_config['ef_range'][0]
                        else:
                            index_params['ef'] = 64
                else:
                    # 使用默认参数
                    if index_type == 'ivfflat':
                        index_params = {
                            "nlist": 100,
                            "nprobe": 16,
                            "metric": "l2"
                        }
                    elif index_type == 'ivfpq':
                        index_params = {
                            "nlist": 100,
                            "m": 8,
                            "nbits": 8,
                            "nprobe": 16,
                            "ivfpq_refine_k_factor": 1,
                            "metric": "l2"
                        }
                    elif index_type == 'hnsw':
                        index_params = {
                            "M": 16,
                            "efConstruction": 200,
                            "ef": 64,
                            "metric": "l2"
                        }
            
            # 确定度量类型
            metric = index_params.get("metric", "l2")
            if metric == "cosine" or metric == "angular":
                distance_metric = "cosine_distance"
            elif metric == "ip" or metric == "inner_product":
                distance_metric = "inner_product"
            else:
                distance_metric = "l2_distance"  # 默认欧氏距离
            
            # 创建索引
            index_name = f"{table_name}_{index_type}_idx"
            start_time = time.time()
            
            try:
                if index_type == "ivfflat":
                    # IVF-Flat索引
                    nlist = index_params.get("nlist", 100)
                    
                    create_index_sql = f"""
                        CREATE INDEX {index_name} ON {table_name} 
                        USING vectors ({vector_column} ivfflat_ops)
                        WITH (clustering_type = ivfflat, distance_metric = {distance_metric}, clustering_params = '{nlist}')
                    """
                    cursor.execute(create_index_sql)
                    
                    index_info = {
                        "index_type": "ivfflat",
                        "nlist": nlist,
                        "metric": metric
                    }
                    
                elif index_type == "ivfpq":
                    # IVF-PQ索引
                    nlist = index_params.get("nlist", 100)
                    m = index_params.get("m", 8)  # 子量化器数量
                    
                    create_index_sql = f"""
                        CREATE INDEX {index_name} ON {table_name} 
                        USING vectors ({vector_column} ivfpq_ops)
                        WITH (
                            clustering_type = ivfpq, 
                            distance_metric = {distance_metric}, 
                            clustering_params = '{nlist}',
                            pq_params = '{m}'
                        )
                    """
                    cursor.execute(create_index_sql)
                    
                    index_info = {
                        "index_type": "ivfpq",
                        "nlist": nlist,
                        "m": m,
                        "metric": metric
                    }
                    
                elif index_type == "hnsw":
                    # HNSW索引
                    m = index_params.get("M", 16)  # 每层最大邻居数
                    ef_construction = index_params.get("efConstruction", 200)  # 构建时的搜索宽度
                    
                    create_index_sql = f"""
                        CREATE INDEX {index_name} ON {table_name} 
                        USING vectors ({vector_column} hnsw_ops)
                        WITH (
                            clustering_type = hnsw, 
                            distance_metric = {distance_metric}, 
                            m = {m},
                            ef_construction = {ef_construction}
                        )
                    """
                    cursor.execute(create_index_sql)
                    
                    index_info = {
                        "index_type": "hnsw",
                        "M": m,
                        "efConstruction": ef_construction,
                        "metric": metric
                    }
                    
                else:
                    raise ValueError(f"不支持的索引类型: {index_type}")
                
                # 计算索引创建时间
                build_time = time.time() - start_time
                logger.info(f"索引 {index_name} 创建成功，耗时 {build_time:.2f} 秒")
                
                # 获取索引大小
                cursor.execute(f"SELECT pg_size_pretty(pg_relation_size('{index_name}'))")
                index_size = cursor.fetchone()[0]
                
                # 返回索引信息
                index_info.update({
                    "index_name": index_name,
                    "build_time": build_time,
                    "index_size": index_size
                })
                
                return index_info
                
            except Exception as e:
                logger.error(f"创建索引失败: {e}")
                raise
        finally:
            cursor.close()
    
    def test_performance(self, table_name: str, query_table: str, topk: int, 
                       index_params: Optional[Dict[str, Any]] = None, concurrency: int = None) -> Dict[str, Any]:
        """
        测试索引性能
        
        Args:
            table_name: 表名
            query_table: 查询表名
            topk: 返回的最近邻数量
            index_params: 索引参数
            concurrency: 并发数
            
        Returns:
            包含性能测试结果的字典
        """
        # 为测试性能创建新的连接
        connection = self.get_connection()
        cursor = connection.cursor()
        
        try:
            # 检查表是否存在
            cursor.execute(f"SELECT 1 FROM pg_tables WHERE tablename = '{table_name}'")
            if not cursor.fetchone():
                raise ValueError(f"表 {table_name} 不存在")
            
            # 检查查询表是否存在
            cursor.execute(f"SELECT 1 FROM pg_tables WHERE tablename = '{query_table}'")
            if not cursor.fetchone():
                raise ValueError(f"查询表 {query_table} 不存在")
            
            # 获取向量字段名
            cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}' AND data_type = 'floatvector'")
            result = cursor.fetchone()
            if not result:
                raise ValueError(f"表 {table_name} 中未找到向量字段")
            
            vector_column = result[0]
            
            # 获取查询表的向量字段名
            cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{query_table}' AND data_type = 'floatvector'")
            result = cursor.fetchone()
            if not result:
                raise ValueError(f"查询表 {query_table} 中未找到向量字段")
            
            query_vector_column = result[0]
            
            # 获取查询表中的向量数量
            cursor.execute(f"SELECT COUNT(*) FROM {query_table}")
            query_count = cursor.fetchone()[0]
            
            if query_count == 0:
                raise ValueError(f"查询表 {query_table} 中没有数据")
            
            # 创建索引
            index_info = self.create_index(table_name, self._find_index_type, index_params)
            
            # 设置并发数
            if concurrency is None:
                concurrency = min(self.parallel_workers, query_count)
            else:
                concurrency = min(concurrency, query_count)
            
            logger.info(f"使用 {concurrency} 个并发线程进行性能测试")
            
            # 获取查询向量
            cursor.execute(f"SELECT id, {query_vector_column} FROM {query_table}")
            query_data = cursor.fetchall()
            
            # 准备查询参数
            query_queue = queue.Queue()
            result_queue = queue.Queue()
            
            for query_id, query_vector in query_data:
                query_queue.put((query_id, query_vector))
            
            # 定义工作线程函数
            def worker(query_id, query_vector):
                # 为每个工作线程创建新的连接
                worker_connection = self.get_connection()
                worker_cursor = worker_connection.cursor()
                
                try:
                    start_time = time.time()
                    
                    # 执行向量搜索
                    search_sql = f"""
                        SELECT id, {vector_column} <-> '{query_vector}' AS distance
                        FROM {table_name}
                        ORDER BY distance
                        LIMIT {topk}
                    """
                    worker_cursor.execute(search_sql)
                    search_results = worker_cursor.fetchall()
                    
                    end_time = time.time()
                    query_time = end_time - start_time
                    
                    # 返回结果
                    result = {
                        "query_id": query_id,
                        "time": query_time,
                        "results": [result[0] for result in search_results]
                    }
                    
                    return result
                except Exception as e:
                    logger.error(f"查询失败: {e}")
                    return {
                        "query_id": query_id,
                        "error": str(e)
                    }
                finally:
                    worker_cursor.close()
                    worker_connection.close()
            
            # 启动工作线程
            threads = []
            results = []
            
            # 使用线程池执行查询
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_to_query = {
                    executor.submit(worker, query_id, query_vector): (query_id, query_vector)
                    for query_id, query_vector in query_data
                }
                
                for future in concurrent.futures.as_completed(future_to_query):
                    query_id, query_vector = future_to_query[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"查询 {query_id} 失败: {e}")
                        results.append({
                            "query_id": query_id,
                            "error": str(e)
                        })
            
            # 计算性能指标
            successful_queries = [r for r in results if "error" not in r]
            
            if not successful_queries:
                raise ValueError("所有查询都失败了")
            
            query_times = [r["time"] for r in successful_queries]
            avg_time = sum(query_times) / len(query_times)
            min_time = min(query_times)
            max_time = max(query_times)
            p50_time = sorted(query_times)[len(query_times) // 2]
            p95_time = sorted(query_times)[int(len(query_times) * 0.95)]
            p99_time = sorted(query_times)[int(len(query_times) * 0.99)]
            
            # 计算QPS
            qps = len(successful_queries) / sum(query_times)
            
            # 计算召回率
            recall_results = self.compute_recall([r["results"] for r in successful_queries], f"{table_name}_groundtruth")
            
            # 返回性能测试结果
            performance_results = {
                "index_info": index_info,
                "query_count": len(successful_queries),
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "p50_time": p50_time,
                "p95_time": p95_time,
                "p99_time": p99_time,
                "qps": qps,
                "recall": recall_results
            }
            
            return performance_results
        finally:
            cursor.close()
            connection.close()
    
    def compute_recall(self, search_results: List[List[int]], groundtruth_table: str) -> Dict[str, Any]:
        """
        计算召回率
        
        Args:
            search_results: 搜索结果，每个查询的结果是一个ID列表
            groundtruth_table: 真实结果表名
            
        Returns:
            包含召回率信息的字典
        """
        # 为计算召回率创建新的连接
        connection = self.get_connection()
        cursor = connection.cursor()
        
        try:
            # 检查真实结果表是否存在
            cursor.execute(f"SELECT 1 FROM pg_tables WHERE tablename = '{groundtruth_table}'")
            if not cursor.fetchone():
                raise ValueError(f"真实结果表 {groundtruth_table} 不存在")
            
            # 获取真实结果
            groundtruth_results = self.get_groundtruth_results(groundtruth_table)
            
            if len(search_results) != len(groundtruth_results):
                raise ValueError(f"搜索结果数量 ({len(search_results)}) 与真实结果数量 ({len(groundtruth_results)}) 不匹配")
            
            # 计算每个查询的召回率
            recalls = []
            for i, (search_result, groundtruth) in enumerate(zip(search_results, groundtruth_results)):
                # 计算交集大小
                intersection = set(search_result) & set(groundtruth)
                recall = len(intersection) / len(groundtruth) if groundtruth else 1.0
                recalls.append(recall)
            
            # 计算平均召回率
            avg_recall = sum(recalls) / len(recalls) if recalls else 0
            
            # 计算各个百分位数
            sorted_recalls = sorted(recalls)
            p50_recall = sorted_recalls[len(sorted_recalls) // 2] if sorted_recalls else 0
            p95_recall = sorted_recalls[int(len(sorted_recalls) * 0.95)] if sorted_recalls else 0
            p99_recall = sorted_recalls[int(len(sorted_recalls) * 0.99)] if sorted_recalls else 0
            
            # 返回召回率信息
            return {
                "avg_recall": avg_recall,
                "min_recall": min(recalls) if recalls else 0,
                "max_recall": max(recalls) if recalls else 0,
                "p50_recall": p50_recall,
                "p95_recall": p95_recall,
                "p99_recall": p99_recall
            }
        finally:
            cursor.close()
            connection.close()
    
    def drop_index(self, table_name: str) -> bool:
        """
        删除表上的所有索引
        
        Args:
            table_name: 表名
            
        Returns:
            是否成功删除索引
        """
        # 为删除索引创建新的连接
        connection = self.get_connection()
        cursor = connection.cursor()
        
        try:
            # 检查表是否存在
            cursor.execute(f"SELECT 1 FROM pg_tables WHERE tablename = '{table_name}'")
            if not cursor.fetchone():
                logger.warning(f"表 {table_name} 不存在")
                return False
            
            # 获取表上的所有索引
            cursor.execute(f"SELECT indexname FROM pg_indexes WHERE tablename = '{table_name}'")
            indexes = cursor.fetchall()
            
            if not indexes:
                logger.info(f"表 {table_name} 上没有索引")
                return True
            
            # 删除所有索引
            for index_name in [row[0] for row in indexes]:
                logger.info(f"删除索引 {index_name}")
                cursor.execute(f"DROP INDEX {index_name}")
            
            logger.info(f"已删除表 {table_name} 上的所有索引")
            return True
        except Exception as e:
            logger.error(f"删除索引失败: {e}")
            return False
        finally:
            cursor.close()
            connection.close()
    
    def close(self) -> None:
        """关闭数据库连接"""
        # 使用engine中的方法关闭连接
        db_engine.close_connection(self.connection, self.cursor)
        self.connection = None
        self.cursor = None
