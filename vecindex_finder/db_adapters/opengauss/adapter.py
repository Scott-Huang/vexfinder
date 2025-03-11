#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
import logging
import io
import psycopg2
import concurrent.futures
import json
import tqdm
from typing import Dict, List, Any, Tuple, Optional
from queue import Queue
import os
import yaml

from ..base import VectorDatabaseAdapter

logger = logging.getLogger(__name__)

class OpenGaussAdapter(VectorDatabaseAdapter):
    """OpenGauss数据库适配器"""
    
    def __init__(self):
        """初始化OpenGauss适配器"""
        self.connection = None
        self.cursor = None
        self.metric_type = None
        self.table_name = None      
        self.dimension = None
        self.vector_column_name = None
        self._sample_table_name = None
        self._query_table_name = None
        self._query_get_type = None
        self._query_data_path = None
        self._find_index_auto = None
        self._find_index_type = None
        self.parallel_workers = 8
        self.config = None
        self.load_and_check_config()
    
    def load_and_check_config(self):
        """
        加载配置文件并检查必要配置项
        
        Raises:
            FileNotFoundError: 如果配置文件不存在
            ValueError: 如果配置文件中缺少必要信息
        """
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'config.yml')
        
        # 检查配置文件是否存在
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件 {config_path} 不存在")
        
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                logger.info(f"成功加载配置文件: {config_path}")
                
                # 检查必要配置项是否存在
                required_sections = ['connection', 'sampling', 'performance']
                for section in required_sections:
                    if section not in self.config:
                        raise ValueError(f"配置文件中缺少必要部分: {section}")
                
                # 检查table_info配置
                required_table_info_fields = ['name', 'vector_column', 'dimension']
                for field in required_table_info_fields:
                    if field not in self.config['table_info']:
                        raise ValueError(f"table_info配置中缺少必要字段: {field}")
                
                self.table_name = self.config['table_info']['table_name']
                self.vector_column_name = self.config['table_info']['vector_column_name']
                self.dimension = self.config['table_info']['dimension']

                self._sample_table_name = f"{self.table_name}_sample_vecindex_finder"
                self._query_table_name = f"{self.table_name}_query_vecindex_finder"

                # 检查连接配置
                required_connection_fields = ['host', 'port', 'user', 'password', 'dbname']
                for field in required_connection_fields:
                    if field not in self.config['connection']:
                        raise ValueError(f"连接配置中缺少必要字段: {field}")
                
                
                # 检查采样配置
                required_sampling_fields = ['default_ratio', 'min_sample_count', 'max_sample_count']
                for field in required_sampling_fields:
                    if field not in self.config['sampling']:
                        raise ValueError(f"采样配置中缺少必要字段: {field}")

                # 检查query配置
                required_query_fields = ['query_get_type', 'query_data_path']
                for field in required_query_fields:
                    if field not in self.config['query']:
                        raise ValueError(f"query配置中缺少必要字段: {field}")

                self._query_get_type = self.config['query']['query_get_type']
                self._query_data_path = self.config['query']['query_data_path']

                # 设置并行工作线程数
                if 'parallel_workers' not in self.config:
                    raise ValueError("配置文件中缺少 parallel_workers 配置")
                self.parallel_workers = self.config['parallel_workers']

                # 检查索引配置
                required_index_fields = ['find_index_type', 'auto']
                for field in required_index_fields:
                    if field not in self.config['index_types']:
                        raise ValueError(f"索引配置中缺少必要字段: {field}")
                
                self._find_index_type = self.config['index_types']['find_index_type']
                self._find_index_auto = self.config['index_types']['auto']

                if not self._find_index_auto:
                    if self._find_index_type not in self.config['index_types']:
                        raise ValueError(f"索引配置中缺少必要字段: {self._find_index_type}")

                    if self._find_index_type == 'ivfflat':
                        required_ivfflat_fields = ['nlist', 'nprobe']
                        for field in required_ivfflat_fields:
                            if field not in self.config['index_types']['ivfflat']:
                                raise ValueError(f"ivfflat配置中缺少必要字段: {field}")
                    elif self._find_index_type == 'ivfpq':
                        required_ivfpq_fields = ['nlist', 'm', 'nbits', 'nprobe', 'ivfpq_refine_k_factor']
                        for field in required_ivfpq_fields:
                            if field not in self.config['index_types']['ivfpq']:
                                raise ValueError(f"ivfpq配置中缺少必要字段: {field}")
                    elif self._find_index_type == 'hnsw':
                        required_hnsw_fields = ['M', 'efConstruction', 'ef']
                        for field in required_hnsw_fields:
                            if field not in self.config['index_types']['hnsw']:
                                raise ValueError(f"hnsw配置中缺少必要字段: {field}")
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    def connect(self):
        """
        连接到OpenGauss服务器
        
        Args:
            config: 连接配置，包含host、port等信息，如果为None则使用配置文件中的配置
            
        Returns:
            self: 返回自身实例以支持链式调用
        """
        # 如果没有提供配置，则使用配置文件中的配置
        if self.config is None:
            raise ValueError("未提供连接配置，且配置文件中没有连接配置")
        
        logger.info(f"连接到OpenGauss服务器: {self.config['connection']['host']}:{self.config['connection']['port']}")
        
        try:
            self.connection = psycopg2.connect(
                host=self.config['connection']['host'],
                port=self.config['connection']['port'],
                user=self.config['connection']['user'],
                password=self.config['connection']['password'],
                dbname=self.config['connection']['dbname']
            )
            self.connection.autocommit = True
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
        
        logger.info(f"从表 {self.table_name} 采样 {sample_count}/{total_count} 条数据 ({sample_count/total_count*100:.2f}%)")
        
        # 检查配置文件中指定的向量列是否存在
        self.cursor.execute(f"""
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = '{self.table_name}' 
            AND column_name = '{self.vector_column_name}'
        """)
        result = self.cursor.fetchone()
        if not result:
            raise ValueError(f"表 {self.table_name} 中未找到配置的向量列 {self.vector_column_name}")
                
        # 创建采样表（不创建索引）
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
    
    def sample_query_data(self, count: int, exclude_ids: Optional[List[int]] = None) -> str:
        """
        采样查询数据并存储到新表中
        
        Args:
            table_name: 表名
            count: 采样数量
            exclude_ids: 需要排除的ID列表
            
        Returns:
            查询表名
        """
        # 检查表是否存在
        self.cursor.execute(f"SELECT 1 FROM pg_tables WHERE tablename = '{table_name}'")
        if not self.cursor.fetchone():
            raise ValueError(f"表 {table_name} 不存在")
        
        # 获取向量字段名
        self.cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}' AND data_type = 'floatvector'")
        result = self.cursor.fetchone()
        if not result:
            raise ValueError(f"表 {table_name} 中未找到向量字段")
        
        vector_column = result[0]
        
        # 获取维度信息
        self.cursor.execute(f"SELECT vector_dimensions(a.{vector_column}) FROM {table_name} a LIMIT 1")
        dimension = self.cursor.fetchone()[0]
        
        # 创建查询表名
        query_table_name = f"{table_name}_query_{int(time.time())}"
        
        # 创建查询表（包含一个用于存储最近邻距离的JSON列）
        self.cursor.execute(f"""
            CREATE TABLE {query_table_name} (
                id SERIAL PRIMARY KEY,
                original_id INTEGER,
                {vector_column} floatvector({dimension}),
                nearest_neighbors JSONB DEFAULT NULL
            )
        """)
        
        logger.info(f"创建查询表 {query_table_name}")
        
        # 构建排除ID的条件
        exclude_condition = ""
        if exclude_ids and len(exclude_ids) > 0:
            # 将大列表分成更小的块，避免SQL参数过长
            max_ids_per_chunk = 1000
            exclude_chunks = [exclude_ids[i:i + max_ids_per_chunk] for i in range(0, len(exclude_ids), max_ids_per_chunk)]
            
            exclude_conditions = []
            for chunk in exclude_chunks:
                ids_str = ','.join(str(id) for id in chunk)
                exclude_conditions.append(f"id NOT IN ({ids_str})")
            
            exclude_condition = " AND ".join(exclude_conditions)
            exclude_condition = f" WHERE {exclude_condition}"
        
        # 采样数据并插入到新表
        query = f"""
            INSERT INTO {query_table_name} (original_id, {vector_column})
            SELECT id, {vector_column} FROM {table_name}{exclude_condition} 
            ORDER BY RANDOM() LIMIT {count}
        """
        self.cursor.execute(query)
        
        # 获取实际采样数量
        self.cursor.execute(f"SELECT COUNT(*) FROM {query_table_name}")
        actual_count = self.cursor.fetchone()[0]
        
        logger.info(f"查询数据采样完成，共插入 {actual_count} 条向量数据到表 {query_table_name}")
        return query_table_name
    
    def compute_nearest_neighbors(self, train_table: str, test_table: str, k: int = 100, metric: str = "l2", concurrency: int = 8) -> None:
        """
        计算测试表中每个向量在训练表中的最近邻，并将结果存储在测试表中
        
        Args:
            train_table: 训练数据表名（采样表）
            test_table: 测试数据表名（查询表）
            k: 最近邻数量
            metric: 距离度量类型 ('l2', 'ip', 'cosine')
            concurrency: 并发线程数
        """
        # 检查表是否存在
        for table in [train_table, test_table]:
            self.cursor.execute(f"SELECT 1 FROM pg_tables WHERE tablename = '{table}'")
            if not self.cursor.fetchone():
                raise ValueError(f"表 {table} 不存在")
        
        # 获取向量字段名
        self.cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{train_table}' AND data_type = 'floatvector'")
        vector_column = self.cursor.fetchone()[0]
        
        # 确定距离操作符
        if metric == "cosine" or metric == "angular":
            distance_operator = "<=>"  # 余弦距离
        else:
            distance_operator = "<->"  # 欧氏距离
        
        # 获取测试表中的所有ID
        self.cursor.execute(f"SELECT id, {vector_column} FROM {test_table}")
        test_data = self.cursor.fetchall()
        
        logger.info(f"开始计算 {len(test_data)} 个查询向量的 {k} 个最近邻")
        
        # 构建查询SQL
        query_sql = f"""
            SELECT original_id, {vector_column} {distance_operator} %s AS distance 
            FROM {train_table} 
            ORDER BY distance ASC 
            LIMIT {k}
        """
        
        # 如果需要并发，使用多线程
        if concurrency > 1:
            results_queue = Queue()
            
            def worker(test_id, test_vector):
                try:
                    # 将向量格式化为OpenGauss接受的格式
                    vector_str = str(test_vector)
                    
                    # 创建单独的数据库连接和游标，避免并发问题
                    conn = psycopg2.connect(
                        host=self.host,
                        port=self.port,
                        user=self.user,
                        password=self.password,
                        dbname=self.dbname
                    )
                    conn.autocommit = True
                    cur = conn.cursor()
                    
                    # 执行查询
                    cur.execute(query_sql, (vector_str,))
                    
                    # 获取结果
                    neighbors = []
                    for row in cur.fetchall():
                        neighbors.append({"id": row[0], "distance": float(row[1])})
                    
                    # 更新测试表
                    neighbors_json = json.dumps(neighbors)
                    update_sql = f"UPDATE {test_table} SET nearest_neighbors = %s WHERE id = %s"
                    cur.execute(update_sql, (neighbors_json, test_id))
                    
                    # 关闭连接
                    cur.close()
                    conn.close()
                    
                    # 将结果放入队列
                    results_queue.put((test_id, len(neighbors)))
                    
                except Exception as e:
                    logger.error(f"并发查询出错: {e}")
                    results_queue.put((test_id, 0))
            
            # 创建线程池
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                # 提交所有任务
                futures = []
                for test_id, test_vector in test_data:
                    futures.append(executor.submit(worker, test_id, test_vector))
                
                # 等待所有任务完成并显示进度
                for _ in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="计算最近邻"):
                    pass
            
            # 收集结果
            success_count = 0
            while not results_queue.empty():
                _, count = results_queue.get()
                if count > 0:
                    success_count += 1
            
            logger.info(f"完成 {success_count}/{len(test_data)} 个查询向量的最近邻计算")
            
        else:
            # 串行查询
            for i, (test_id, test_vector) in enumerate(test_data):
                # 将向量格式化为OpenGauss接受的格式
                vector_str = str(test_vector)
                
                # 执行查询
                self.cursor.execute(query_sql, (vector_str,))
                
                # 获取结果
                neighbors = []
                for row in self.cursor.fetchall():
                    neighbors.append({"id": row[0], "distance": float(row[1])})
                
                # 更新测试表
                neighbors_json = json.dumps(neighbors)
                update_sql = f"UPDATE {test_table} SET nearest_neighbors = %s WHERE id = %s"
                self.cursor.execute(update_sql, (neighbors_json, test_id))
                
                if i % 10 == 0:
                    logger.info(f"已完成 {i}/{len(test_data)} 个查询向量的最近邻计算")
        
        logger.info(f"所有查询向量的最近邻计算完成")
    
    def get_groundtruth_results(self, test_table: str) -> List[List[int]]:
        """
        从测试表中获取精确搜索结果
        
        Args:
            test_table: 测试表名
            
        Returns:
            每个查询的最近邻ID列表
        """
        # 检查表是否存在
        self.cursor.execute(f"SELECT 1 FROM pg_tables WHERE tablename = '{test_table}'")
        if not self.cursor.fetchone():
            raise ValueError(f"表 {test_table} 不存在")
        
        # 获取所有查询的最近邻
        self.cursor.execute(f"SELECT id, nearest_neighbors FROM {test_table} ORDER BY id")
        rows = self.cursor.fetchall()
        
        # 解析结果
        results = []
        
        for _, neighbors_json in rows:
            if neighbors_json:
                neighbors = json.loads(neighbors_json)
                # 提取ID
                neighbor_ids = [n["id"] for n in neighbors]
                results.append(neighbor_ids)
            else:
                results.append([])
        
        return results
    
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
        在表上创建索引
        
        Args:
            table_name: 表名
            index_type: 索引类型 ('ivfflat', 'ivfpq', 'hnsw')
            index_params: 索引参数，如果为None则使用配置文件中的默认参数
            
        Returns:
            包含索引信息的字典
        """
        # 检查表是否存在
        self.cursor.execute(f"SELECT 1 FROM pg_tables WHERE tablename = '{table_name}'")
        if not self.cursor.fetchone():
            raise ValueError(f"表 {table_name} 不存在")
        
        # 获取向量字段名
        self.cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}' AND data_type = 'floatvector'")
        result = self.cursor.fetchone()
        if not result:
            raise ValueError(f"表 {table_name} 中未找到向量字段")
        
        vector_column = result[0]
        
        # 获取维度信息
        self.cursor.execute(f"SELECT vector_dimensions(a.{vector_column}) FROM {table_name} a LIMIT 1")
        dimension = self.cursor.fetchone()[0]
        
        # 检查是否已有索引
        self.cursor.execute(f"SELECT indexname FROM pg_indexes WHERE tablename = '{table_name}'")
        existing_indexes = self.cursor.fetchall()
        
        if existing_indexes:
            logger.warning(f"表 {table_name} 已有索引，将先删除")
            for index_name in [row[0] for row in existing_indexes]:
                self.cursor.execute(f"DROP INDEX {index_name}")
        
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
                self.cursor.execute(create_index_sql)
                
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
                self.cursor.execute(create_index_sql)
                
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
                self.cursor.execute(create_index_sql)
                
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
            self.cursor.execute(f"SELECT pg_size_pretty(pg_relation_size('{index_name}'))")
            index_size = self.cursor.fetchone()[0]
            
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
    
    def test_performance(self, table_name: str, query_table: str, topk: int, 
                       index_params: Optional[Dict[str, Any]] = None, concurrency: int = None) -> Dict[str, Any]:
        """
        测试索引性能
        
        Args:
            table_name: 原始表名
            query_table: 查询表名
            topk: 查询返回的邻居数量
            index_params: 索引参数，如果为None则使用配置文件中的默认参数
            concurrency: 并发查询数，如果为None则使用配置文件中的并行工作线程数
            
        Returns:
            包含性能指标的字典
        """
        # 检查表是否存在
        for table in [table_name, query_table]:
            self.cursor.execute(f"SELECT 1 FROM pg_tables WHERE tablename = '{table}'")
            if not self.cursor.fetchone():
                raise ValueError(f"表 {table} 不存在")
        
        # 获取索引信息
        self.cursor.execute(
            f"SELECT indexname, indexdef FROM pg_indexes WHERE tablename = '{table_name}'"
        )
        indexes = self.cursor.fetchall()
        
        if not indexes:
            raise ValueError(f"表 {table_name} 没有索引")
        
        # 检测索引类型
        index_name = indexes[0][0]
        index_def = indexes[0][1]
        
        # 如果没有提供索引参数，则使用配置文件中的默认参数
        if index_params is None:
            index_params = {}
        
        # 如果没有提供并发数，则使用配置文件中的并行工作线程数
        if concurrency is None:
            concurrency = self.parallel_workers
        
        # 根据索引定义确定索引类型和参数
        if "ivfflat" in index_def.lower():
            index_type = "ivfflat"
            # 设置nprobe参数
            if 'nprobe' not in index_params and self.config and 'index_types' in self.config and 'ivfflat' in self.config['index_types']:
                if 'nprobe_range' in self.config['index_types']['ivfflat'] and self.config['index_types']['ivfflat']['nprobe_range']:
                    index_params['nprobe'] = self.config['index_types']['ivfflat']['nprobe_range'][0]
            
            nprobe = index_params.get("nprobe", 16)
            self.cursor.execute(f"SET enable_seqscan TO off; SET ivf_probes = {nprobe}")
            search_params = {"nprobe": nprobe}
        elif "ivfpq" in index_def.lower():
            index_type = "ivfpq"
            # 设置IVF-PQ参数
            if 'nprobe' not in index_params and self.config and 'index_types' in self.config and 'ivfpq' in self.config['index_types']:
                if 'nprobe_range' in self.config['index_types']['ivfpq'] and self.config['index_types']['ivfpq']['nprobe_range']:
                    index_params['nprobe'] = self.config['index_types']['ivfpq']['nprobe_range'][0]
            
            if 'ivfpq_refine_k_factor' not in index_params and self.config and 'index_types' in self.config and 'ivfpq' in self.config['index_types']:
                if 'ivfpq_refine_k_factor_range' in self.config['index_types']['ivfpq'] and self.config['index_types']['ivfpq']['ivfpq_refine_k_factor_range']:
                    index_params['ivfpq_refine_k_factor'] = self.config['index_types']['ivfpq']['ivfpq_refine_k_factor_range'][0]
            
            nprobe = index_params.get("nprobe", 16)
            n_factor = index_params.get("ivfpq_refine_k_factor", 1)
            self.cursor.execute(
                f"SET enable_seqscan TO off; SET ivf_probes = {nprobe}; SET ivfpq_refine_k_factor = {n_factor}"
            )
            search_params = {"nprobe": nprobe, "ivfpq_refine_k_factor": n_factor}
        elif "hnsw" in index_def.lower():
            index_type = "hnsw"
            # 设置HNSW参数
            if 'ef' not in index_params and self.config and 'index_types' in self.config and 'hnsw' in self.config['index_types']:
                if 'ef_range' in self.config['index_types']['hnsw'] and self.config['index_types']['hnsw']['ef_range']:
                    index_params['ef'] = self.config['index_types']['hnsw']['ef_range'][0]
            
            ef_search = index_params.get("ef", 64)
            self.cursor.execute(f"SET enable_seqscan TO off; SET hnsw_ef_search = {ef_search}")
            search_params = {"ef": ef_search}
        else:
            raise ValueError(f"未识别的索引类型: {index_def}")
        
        # 获取向量字段名
        self.cursor.execute(
            f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}' AND data_type = 'floatvector'"
        )
        vector_column = self.cursor.fetchone()[0]
        
        # 检测度量类型
        metric = index_params.get("metric", "l2")
        if metric == "cosine" or metric == "angular":
            # 余弦距离
            distance_operator = "<=>"
        else:
            # 欧氏距离
            distance_operator = "<->"
        
        # 获取查询表中的所有向量
        self.cursor.execute(f"SELECT id, {vector_column} FROM {query_table}")
        query_data = self.cursor.fetchall()
        
        # 构建查询SQL
        query_sql = f"SELECT id FROM {table_name} ORDER BY {vector_column} {distance_operator} %s LIMIT {topk}"
        
        # 执行并计时
        start_time = time.time()
        result_ids = []
        
        # 如果需要并发，使用多线程
        if concurrency > 1:
            import threading
            from queue import Queue
            import json
            
            results_queue = Queue()
            
            def worker(query_id, query_vector):
                try:
                    # 将向量格式化为OpenGauss接受的格式
                    vector_str = str(query_vector)
                    
                    # 创建单独的数据库连接和游标，避免并发问题
                    conn = psycopg2.connect(
                        host=self.host,
                        port=self.port,
                        user=self.user,
                        password=self.password,
                        dbname=self.dbname
                    )
                    conn.autocommit = True
                    cur = conn.cursor()
                    
                    # 设置必要的参数
                    if index_type == "ivfflat":
                        cur.execute(f"SET enable_seqscan TO off; SET ivf_probes = {search_params['nprobe']}")
                    elif index_type == "ivfpq":
                        cur.execute(
                            f"SET enable_seqscan TO off; SET ivf_probes = {search_params['nprobe']}; "
                            f"SET ivfpq_refine_k_factor = {search_params['ivfpq_refine_k_factor']}"
                        )
                    elif index_type == "hnsw":
                        cur.execute(f"SET enable_seqscan TO off; SET hnsw_ef_search = {search_params['ef']}")
                    
                    # 执行查询
                    cur.execute(query_sql, (vector_str,))
                    
                    # 获取结果
                    ids = [row[0] for row in cur.fetchall()]
                    
                    # 关闭连接
                    cur.close()
                    conn.close()
                    
                    # 将结果放入队列
                    results_queue.put((query_id, ids))
                    
                except Exception as e:
                    logger.error(f"并发查询出错: {e}")
                    results_queue.put((query_id, []))
            
            # 创建线程池
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                # 提交所有任务
                futures = []
                for query_id, query_vector in query_data:
                    futures.append(executor.submit(worker, query_id, query_vector))
            
            # 等待所有任务完成并显示进度
            for _ in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="执行索引查询"):
                pass
            
            # 收集结果
            results = []
            while not results_queue.empty():
                results.append(results_queue.get())
            
            # 按查询ID排序结果
            results.sort(key=lambda x: x[0])
            result_ids = [r[1] for r in results]
            
        else:
            # 串行查询
            for i, (query_id, query_vector) in enumerate(query_data):
                # 将向量格式化为OpenGauss接受的格式
                vector_str = str(query_vector)
                
                # 执行查询
                self.cursor.execute(query_sql, (vector_str,))
                
                # 获取结果
                ids = [row[0] for row in self.cursor.fetchall()]
                result_ids.append(ids)
                
                if i % 10 == 0:
                    logger.debug(f"已完成 {i}/{len(query_data)} 个查询")
        
        # 查询完成，计算性能指标
        total_time = time.time() - start_time
        qps = len(query_data) / total_time
        latency = total_time / len(query_data) * 1000  # 毫秒
        
        logger.info(f"性能测试完成，QPS: {qps:.2f}，平均延迟: {latency:.2f}ms")
        
        # 返回结果
        return {
            "qps": qps,
            "latency": latency,
            "result_ids": result_ids,
            "index_type": index_type,
            "search_params": search_params,
            "total_time": total_time
        }
    
    def compute_recall(self, search_results: List[List[int]], groundtruth_table: str) -> Dict[str, Any]:
        """
        计算召回率
        
        Args:
            search_results: 索引搜索结果，每个查询的topk结果ID
            groundtruth_table: 包含精确搜索结果的表名
            
        Returns:
            包含召回率和评估信息的字典
        """
        # 获取精确搜索结果
        groundtruth_results = self.get_groundtruth_results(groundtruth_table)
        
        if len(search_results) != len(groundtruth_results):
            raise ValueError(f"搜索结果数量 ({len(search_results)}) 与精确结果数量 ({len(groundtruth_results)}) 不匹配")
            
        total_queries = len(search_results)
        recall_sum = 0.0
        
        # 获取配置中的召回率阈值
        min_recall = 0.8
        max_recall = 0.999
        if self.config and 'performance' in self.config:
            min_recall = self.config['performance'].get('min_recall', 0.8)
            max_recall = self.config['performance'].get('max_recall', 0.999)
        
        # 计算每个查询的召回率
        query_recalls = []
        for i in range(total_queries):
            # 计算结果重叠
            gt_set = set(groundtruth_results[i])
            result_set = set(search_results[i])
            overlap = len(gt_set.intersection(result_set))
            
            # 计算单个查询的召回率
            single_recall = overlap / len(gt_set) if len(gt_set) > 0 else 0
            query_recalls.append(single_recall)
            recall_sum += single_recall
            
        # 计算平均召回率
        avg_recall = recall_sum / total_queries
        
        # 评估召回率
        recall_status = "good"
        if avg_recall < min_recall:
            recall_status = "low"
        elif avg_recall > max_recall:
            recall_status = "excellent"
        
        logger.info(f"平均召回率: {avg_recall:.4f}, 状态: {recall_status}")
        
        # 返回结果
        return {
            "recall": avg_recall,
            "status": recall_status,
            "min_recall": min_recall,
            "max_recall": max_recall,
            "query_recalls": query_recalls
        }
    
    def drop_index(self, table_name: str) -> bool:
        """
        删除索引和临时表
        
        Args:
            table_name: 表名
            
        Returns:
            操作是否成功
        """
        try:
            # 检查表是否存在
            self.cursor.execute(f"SELECT 1 FROM pg_tables WHERE tablename = '{table_name}'")
            if not self.cursor.fetchone():
                logger.warning(f"表 {table_name} 不存在，无需删除")
                return True
            
            # 获取该表的所有索引
            self.cursor.execute(f"SELECT indexname FROM pg_indexes WHERE tablename = '{table_name}'")
            indexes = [row[0] for row in self.cursor.fetchall()]
            
            # 删除所有索引
            for index_name in indexes:
                logger.info(f"删除索引 {index_name}")
                self.cursor.execute(f"DROP INDEX IF EXISTS {index_name}")
            
            # 删除表
            logger.info(f"删除表 {table_name}")
            self.cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            return True
        except Exception as e:
            logger.error(f"删除索引和表失败: {e}")
            return False
    
    def close(self) -> None:
        """关闭数据库连接"""
        if self.cursor:
            self.cursor.close()
        
        if self.connection:
            logger.info("关闭OpenGauss连接")
            self.connection.close()
            self.connection = None
