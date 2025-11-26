from __future__ import absolute_import

import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing


def knn_threshold(data, count, epsilon):
    return data[count - 1] + epsilon


def get_recall_values(distances, run_distances, count, epsilon=1e-3):
    """
    通过比较ground truth距离和索引搜索距离来计算召回率
    
    参数:
        distances: 精确KNN计算得到的ground truth距离数组
        run_distances: 向量索引搜索得到的距离数组
        count: 要考虑的最近邻数量 (k)
        epsilon: 用于阈值计算的小值
        
    返回:
        包含平均召回率
    """
    # 定义一个处理单个查询项的函数
    def process_item(idx):
        # 为当前查询计算阈值 (使用ground truth距离)
        t = knn_threshold(distances[idx], count, epsilon)
        
        # 计算在阈值内的结果数量
        actual = 0
        for d in run_distances[idx][:count]:
            if d <= t:
                actual += 1
                
        return actual / float(count)  # 返回召回率 (0-1之间)

    recalls = np.zeros(len(run_distances))
    
    # 确定使用的线程数
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    
    # 使用ThreadPoolExecutor进行并行处理
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 并行执行任务，传入索引值而不是数据本身
        results = list(executor.map(process_item, range(len(run_distances))))
        
        # 将结果填入recalls数组
        for i, recall in enumerate(results):
            recalls[i] = recall
    
    return np.mean(recalls)