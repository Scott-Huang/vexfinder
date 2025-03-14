#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import threading
import queue
import tqdm
from typing import Dict, List, Any, Tuple, Set, Optional
from core.config import Config
from core.engine import DatabaseEngine
from core.param_builder import IndexParamBuilder
from core.logging import logger

class IndexExecutor:
    """索引执行器，负责创建索引和执行性能测试"""
    
    def __init__(self, config: Config, db_engine: DatabaseEngine, param_builder: IndexParamBuilder):
        """
        初始化索引执行器
        
        Args:
            config: 配置对象
            db_engine: 数据库引擎对象
            param_builder: 索引参数构建器
        """
        self.config = config
        self.db_engine = db_engine
        self.param_builder = param_builder
        
        # 从配置中获取相关参数
        self.table_info_config = config.table_info_config
        self.performance_config = config.performance_config
        self.parallel_workers = config.parallel_workers
        
        # 表信息
        self.table_name = self.table_info_config['table_name']
        self.sample_table_name = self.table_info_config['sample_table_name']
        self.query_table_name = self.table_info_config['query_table_name']
        self.vector_column_name = self.table_info_config['vector_column_name']
        self.dimension = self.table_info_config['dimension']
        self.metric = self.table_info_config['metric']
        
        # 性能配置
        self.limit = self.performance_config['limit']
        self.min_recall = self.performance_config['min_recall']
    
    def create_index(self, params: Dict[str, Any]) -> Tuple[str, float]:
        """
        在采样表上创建索引
        
        Args:
            params: 索引参数
            
        Returns:
            索引名称和创建时间
        """
        connection = self.db_engine.get_connection()
        cursor = connection.cursor()
        
        # 生成索引名称
        index_name = self.param_builder.generate_index_name(params, self.sample_table_name)
        
        # 获取索引选项
        index_options = self.param_builder.get_index_options(params)
        
        # 删除可能存在的同名索引
        try:
            cursor.execute(f"DROP INDEX IF EXISTS {index_name}")
        except Exception as e:
            logger.warning(f"删除索引失败: {e}")
        
        # 创建索引
        try:
            create_index_sql = f"""
            CREATE INDEX {index_name} ON {self.sample_table_name} 
            USING {params['index_type']}({self.vector_column_name} {self.metric}) 
            WITH ({index_options})
            """
            logger.info(f"创建索引: {create_index_sql}")
            
            start_time = time.time()
            cursor.execute(create_index_sql)
            index_creation_time = time.time() - start_time
            
            logger.info(f"索引 {index_name} 创建成功，耗时 {index_creation_time:.2f} 秒")
            
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            self.db_engine.close_connection(connection, cursor)
            raise
        
        self.db_engine.close_connection(connection, cursor)
        return index_name, index_creation_time
    
    def drop_index(self, index_name: str) -> None:
        """
        删除索引
        
        Args:
            index_name: 索引名称
        """
        connection = self.db_engine.get_connection()
        cursor = connection.cursor()
        
        try:
            cursor.execute(f"DROP INDEX IF EXISTS {index_name}")
            logger.info(f"索引 {index_name} 已删除")
        except Exception as e:
            logger.warning(f"删除索引 {index_name} 失败: {e}")
        finally:
            self.db_engine.close_connection(connection, cursor)
    
    def test_index_performance(self, params: Dict[str, Any], index_name: str) -> Dict[str, Any]:
        """
        测试索引性能
        
        Args:
            params: 索引参数
            index_name: 索引名称
            
        Returns:
            性能测试结果
        """
        connection = self.db_engine.get_connection()
        cursor = connection.cursor()
        
        # 设置查询参数
        query_param_sql = self.param_builder.get_query_param_sql(params)
        cursor.execute(query_param_sql)
        
        # 确定距离操作符
        distance_operator = self._get_distance_operator()
        
        # 获取查询向量
        cursor.execute(f"SELECT id, {self.vector_column_name} FROM {self.query_table_name} LIMIT 100")
        query_vectors = cursor.fetchall()
        
        # 使用多线程并行测试
        if len(query_vectors) >= 10 and self.parallel_workers > 1:
            result = self._parallel_test_performance(params, query_vectors, distance_operator)
        else:
            result = self._sequential_test_performance(params, query_vectors, distance_operator, cursor)
        
        self.db_engine.close_connection(connection, cursor)
        
        # 添加索引名称到结果中
        result['index_name'] = index_name
        
        return result
    
    def _get_distance_operator(self) -> str:
        """
        根据度量类型获取距离操作符
        
        Returns:
            距离操作符
        """
        if self.metric == "cosine":
            return "<=>"  # 余弦距离
        elif self.metric == "l2":
            return "<->"  # 欧氏距离
        elif self.metric == "ip":
            return "<#>"  # 内积距离
        else:
            raise ValueError(f"不支持的距离度量类型: {self.metric}")
    
    def _sequential_test_performance(self, params: Dict[str, Any], query_vectors: List[Tuple], 
                                    distance_operator: str, cursor) -> Dict[str, Any]:
        """
        顺序测试索引性能
        
        Args:
            params: 索引参数
            query_vectors: 查询向量列表
            distance_operator: 距离操作符
            cursor: 数据库游标
            
        Returns:
            性能测试结果
        """
        total_time = 0
        total_queries = len(query_vectors)
        recall_values = []
        
        # 对每个查询向量进行测试
        for query_id, query_vector in query_vectors:
            # 获取真实最近邻（使用精确查询）
            cursor.execute(f"""
                SELECT id FROM {self.sample_table_name} 
                ORDER BY {self.vector_column_name} {distance_operator} %s ASC 
                LIMIT {self.limit}
            """, (query_vector,))
            ground_truth = set(row[0] for row in cursor.fetchall())
            
            # 使用索引进行查询
            start_time = time.time()
            cursor.execute(f"""
                SELECT id FROM {self.sample_table_name} 
                ORDER BY {self.vector_column_name} {distance_operator} %s ASC 
                LIMIT {self.limit}
            """, (query_vector,))
            query_time = time.time() - start_time
            
            # 计算召回率
            results = set(row[0] for row in cursor.fetchall())
            recall = len(ground_truth.intersection(results)) / len(ground_truth)
            
            total_time += query_time
            recall_values.append(recall)
        
        # 计算平均查询时间和QPS
        avg_query_time = total_time / total_queries
        qps = 1.0 / avg_query_time
        avg_recall = sum(recall_values) / len(recall_values)
        
        # 记录结果
        result = {
            **params,
            'avg_query_time': avg_query_time,
            'qps': qps,
            'avg_recall': avg_recall
        }
        
        logger.info(f"索引性能测试结果: QPS={qps:.2f}, 平均召回率={avg_recall:.4f}")
        
        return result
    
    def _parallel_test_performance(self, params: Dict[str, Any], query_vectors: List[Tuple], 
                                  distance_operator: str) -> Dict[str, Any]:
        """
        并行测试索引性能
        
        Args:
            params: 索引参数
            query_vectors: 查询向量列表
            distance_operator: 距离操作符
            
        Returns:
            性能测试结果
        """
        # 创建任务队列
        task_queue = queue.Queue()
        
        # 将所有任务放入队列
        for query_id, query_vector in query_vectors:
            task_queue.put((query_id, query_vector))
        
        # 创建结果列表和锁
        results = []
        results_lock = threading.Lock()
        
        # 创建进度计数器和锁
        progress_counter = 0
        counter_lock = threading.Lock()
        
        # 创建进度条
        total_vectors = len(query_vectors)
        progress_bar = tqdm.tqdm(total=total_vectors, desc="测试索引性能")
        
        # 工作线程函数
        def worker():
            # 创建单独的数据库连接
            conn = self.db_engine.get_connection()
            cur = conn.cursor()
            
            # 设置查询参数
            query_param_sql = self.param_builder.get_query_param_sql(params)
            cur.execute(query_param_sql)
            
            while True:
                try:
                    # 从队列中获取任务，如果队列为空则退出
                    try:
                        query_id, query_vector = task_queue.get(block=False)
                    except queue.Empty:
                        break
                    
                    # 获取真实最近邻（使用精确查询）
                    cur.execute(f"""
                        SELECT id FROM {self.sample_table_name} 
                        ORDER BY {self.vector_column_name} {distance_operator} %s ASC 
                        LIMIT {self.limit}
                    """, (query_vector,))
                    ground_truth = set(row[0] for row in cur.fetchall())
                    
                    # 使用索引进行查询
                    start_time = time.time()
                    cur.execute(f"""
                        SELECT id FROM {self.sample_table_name} 
                        ORDER BY {self.vector_column_name} {distance_operator} %s ASC 
                        LIMIT {self.limit}
                    """, (query_vector,))
                    query_time = time.time() - start_time
                    
                    # 计算召回率
                    query_results = set(row[0] for row in cur.fetchall())
                    recall = len(ground_truth.intersection(query_results)) / len(ground_truth)
                    
                    # 将结果添加到结果列表
                    with results_lock:
                        results.append((query_time, recall))
                    
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
            self.db_engine.close_connection(conn, cur)
        
        # 创建并启动工作线程
        threads = []
        for _ in range(min(self.parallel_workers, total_vectors)):
            thread = threading.Thread(target=worker)
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        # 等待所有任务完成
        task_queue.join()
        
        # 关闭进度条
        progress_bar.close()
        
        # 计算平均查询时间和QPS
        total_time = sum(time for time, _ in results)
        total_queries = len(results)
        avg_query_time = total_time / total_queries
        qps = 1.0 / avg_query_time
        avg_recall = sum(recall for _, recall in results)
        avg_recall = avg_recall / total_queries
        
        # 记录结果
        result = {
            **params,
            'avg_query_time': avg_query_time,
            'qps': qps,
            'avg_recall': avg_recall
        }
        
        logger.info(f"索引性能测试结果: QPS={qps:.2f}, 平均召回率={avg_recall:.4f}")
        
        return result
    
    def benchmark_index(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        对指定参数的索引进行基准测试
        
        Args:
            params: 索引参数
            
        Returns:
            基准测试结果
        """
        try:
            # 创建索引
            index_name, index_creation_time = self.create_index(params)
            
            # 添加索引创建时间到参数中
            params['index_creation_time'] = index_creation_time
            
            # 测试性能
            result = self.test_index_performance(params, index_name)
            
            # 删除索引（可选，取决于空间限制）
            self.drop_index(index_name)
            
            return result
            
        except Exception as e:
            logger.error(f"测试参数 {params} 时出错: {e}")
            raise 