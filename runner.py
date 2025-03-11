#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import time
import glob
import yaml
import numpy
import multiprocessing as mp
from functools import partial
from ann_datasets import DATASETS, get_dataset
from distance import metrics
from ann_results import store_results
from definitions import instantiate_algorithm, get_definitions


def calculate_distances(query_vector, candidates, train_data, distance):
    """
    计算查询向量与候选项之间的距离
    
    Args:
        query_vector: 查询向量
        candidates: 候选项索引列表
        train_data: 训练数据
        distance: 距离度量类型
        
    Returns:
        包含(候选项索引, 距离)元组的列表
    """
    # 使用NumPy的向量化操作批量计算距离
    candidate_vectors = numpy.array([train_data[cand_idx] for cand_idx in candidates])
    
    if distance == 'euclidean':
        # 欧氏距离的向量化计算
        distances = numpy.sqrt(numpy.sum((candidate_vectors - query_vector)**2, axis=1))
    elif distance == 'angular':
        # 角度距离的向量化计算
        distances = 1 - numpy.dot(candidate_vectors, query_vector) / (
            numpy.linalg.norm(candidate_vectors, axis=1) * numpy.linalg.norm(query_vector)
        )
    elif distance == 'inner_product' or distance == 'ip':
        # 内积距离的向量化计算
        # 注意：内积越大表示越相似，所以这里取负值作为距离
        distances = -numpy.dot(candidate_vectors, query_vector)
    else:
        # 其他距离度量仍然使用单个计算
        distances = [float(metrics[distance].distance(query_vector, train_data[cand_idx])) for cand_idx in candidates]
    
    # 构建结果列表
    return [(int(candidates[i]), float(distances[i])) for i in range(len(candidates))]


# 将嵌套函数移到外部，使其可以被pickle
def single_query(algo, v, count):
    """仅执行查询并返回原始结果"""
    start = time.time()
    candidates = algo.query(v, count)
    query_time = time.time() - start

    # 这里不再直接访问train_data，而是返回索引，让主进程计算距离
    return (query_time, candidates)


# 修改进程查询函数，使其创建自己的算法实例，不传递h5py对象
def process_queries(process_id, definition, X_subset, dataset_name, distance, count, max_time, query_arguments=None):
    """每个进程创建自己的算法实例并执行查询"""
    # 获取数据集，但只使用必要的部分
    dataset, _, _ = load_and_transform_dataset(dataset_name)
    
    # 创建新的算法实例
    algo = instantiate_algorithm(definition)
    
    # 如果有查询参数，设置它们
    if query_arguments:
        algo.set_query_arguments(*query_arguments)
    
    query_results = []        
    end_time = time.time() + max_time
    current_idx = 0
    
    while time.time() < end_time:
        idx = current_idx % len(X_subset)
        x = X_subset[idx]
        query_time, candidates = single_query(algo, x, count)
        query_results.append((query_time, (candidates, idx)))
        current_idx += 1
    
    return query_results


def run_individual_query(algo, dataset, X_test, distance, count, reads=1, duration=0, definition=None, query_arguments=None, dataset_name=None, calculate_distance=False):
    train_data = dataset["train"]  # h5py Dataset
    
    # 默认使用持续时间模式，如果未设置则使用默认的60秒
    duration = duration if duration > 0 else 60
    print(f"Running test for {duration} seconds...")
    
    all_results = []
    
    if reads > 1:
        # 多进程查询 - 每个进程创建自己的算法实例
        pool = mp.Pool(processes=reads)
        
        # 所有进程同时运行指定的时间
        results_list = pool.map(
            partial(process_queries, 
                   definition=definition,
                   X_subset=X_test, 
                   dataset_name=dataset_name,  # 传递数据集名称而不是h5py对象
                   distance=distance, 
                   count=count, 
                   max_time=duration,
                   query_arguments=query_arguments), 
            range(reads)
        )
            
        pool.close()
        pool.join()
        
        # 合并所有进程的结果
        for proc_results in results_list:
            # 在主进程中计算距离
            processed_results = []
            for query_time, (candidates, idx) in proc_results:
                # calculate_distance 用于控制是否计算实际距离, 当runs>1时，只保留最后一次的运行结果, 所以前面几次的运行结果不计算距离
                if calculate_distance:
                    # 计算实际距离 - 优化版本
                    # 使用NumPy的向量化操作批量计算距离
                    query_vector = X_test[idx]
                    candidates_with_distance = calculate_distances(query_vector, candidates, train_data, distance)
                    processed_results.append((query_time, (candidates_with_distance, idx)))
                else:
                    processed_results.append((query_time, ([], idx)))
            all_results.extend(processed_results)
    else:
        # 单进程查询 - 使用现有算法实例
        all_results = []
        end_time = time.time() + duration
        current_idx = 0
        
        while time.time() < end_time:
            idx = current_idx % len(X_test)
            x = X_test[idx]
            query_time, candidates = single_query(algo, x, count)
            
            # calculate_distance 用于控制是否计算实际距离, 当runs>1时，只保留最后一次的运行结果, 所以前面几次的运行结果不计算距离
            if calculate_distance:
                # 计算实际距离 - 优化版本
                # 使用NumPy的向量化操作批量计算距离
                query_vector = x
                candidates_with_distance = calculate_distances(query_vector, candidates, train_data, distance)
            else:
                candidates_with_distance = []
            
            all_results.append((query_time, (candidates_with_distance, idx)))
            current_idx += 1
    
    # 获取实际执行的查询数
    queries_count = len(all_results)
    print(f"实际执行的查询数: {queries_count}")

    # 总查询时间
    total_query_time = sum(time for time, _ in all_results)

    # 计算平均查询时间
    avg_query_time = total_query_time / queries_count
    print(f"平均查询时间: {avg_query_time:.6f} 秒")

    # 最小查询时间
    best_query_time = min(time for time, _ in all_results)
    print(f"最小查询时间: {best_query_time:.6f} 秒")

    # 最大查询时间
    max_query_time = max(time for time, _ in all_results)
    print(f"最大查询时间: {max_query_time:.6f} 秒")

    # 数据库查询时间 (考虑并发)
    best_search_time = total_query_time / queries_count / reads
    
    print(f"QPS = {1 / best_search_time:.4f}\n")
    
    total_candidates = sum(len(candidates[0]) for _, candidates in all_results)
    avg_candidates = total_candidates / queries_count

    attrs = {
        "name": str(algo),
        "best_search_time": best_search_time,
        "avg_query_time": avg_query_time,
        "min_query_time": best_query_time,
        "max_query_time": max_query_time,
        "queries_count": queries_count,
        "candidates": avg_candidates,
        "distance": distance,
        "count": int(count),
        "reads": reads,
        "duration": duration,
        "total_queries": queries_count
    }
    
    additional = algo.get_additional()
    for k in additional:
        attrs[k] = additional[k]
    return (attrs, all_results)


def load_and_transform_dataset(dataset_name):
    """只返回 HDF5 dataset 对象和测试数据"""
    D, _ = get_dataset(dataset_name)
    X_test = numpy.array(D["test"])  # 测试集通常较小,可以完整加载
    distance = D.attrs["distance"]
    
    return D, X_test, distance


def copy_and_create_index(algo, dataset, run_copy):
    """Builds the ANN index."""
    copy_time = 0
    if run_copy:
        algo.drop()
        t0 = time.time()
        algo.copy(dataset)  # 现在传入的是 dataset 而不是 numpy array
        copy_time = time.time() - t0
    print(f"复制时间: {copy_time:.2f} 秒")
    t0 = time.time()
    algo.create_index()
    create_index_time = time.time() - t0
    print(f"创建索引时间: {create_index_time:.2f} 秒")
    build_time = copy_time + create_index_time
    print(f"构建时间: {build_time:.2f} 秒")
    index_size = algo.get_memory_usage()
    table_size = algo.get_table_usage()

    print(f"Index size: {index_size/1024:0.2f} MB")
    print(f"Table size: {table_size/1024:0.2f} MB")

    return copy_time, create_index_time, build_time, index_size, table_size


def run(definition, dataset_name, count, runs, parallel_workers, run_copy, reads=1, duration=0, drop_after_test=False):
    """Run the algorithm benchmarking."""
    # 获取 HDF5 dataset
    dataset, X_test, distance = load_and_transform_dataset(dataset_name)
    
    algo = instantiate_algorithm(definition)
    if hasattr(algo, "parallel_workers"):
        algo.parallel_workers = parallel_workers
        
    try:
        # 传入 dataset 而不是完整的训练数据
        copy_time, create_index_time, build_time, index_size, table_size = copy_and_create_index(algo, dataset, run_copy)
        
        # 确保至少有一组查询参数
        query_argument_groups = definition.query_argument_groups or [[]]

        for pos, query_arguments in enumerate(query_argument_groups, 1):
            print(f"\n运行第 {pos}/{len(query_argument_groups)}个参数组: {query_arguments}")
            if query_arguments:
                algo.set_query_arguments(*query_arguments)
            
            descriptor, results = [], []
            for run in range(runs):
                print(f"运行第 {run+1} 次, 总共 {runs} 次")
                if run == runs - 1:
                    calculate_distance = True
                else:
                    calculate_distance = False
                descriptor, results = run_individual_query(
                    algo, dataset, X_test, distance, count, reads, duration,
                    definition=definition, query_arguments=query_arguments, dataset_name=dataset_name, calculate_distance=calculate_distance
                )
            descriptor.update({
                "copy_time": copy_time,
                "create_index_time": create_index_time,
                "build_time": build_time,
                "index_size": index_size,
                "table_size": table_size,
                "algo": definition.algorithm,
                "dataset": dataset_name
            })

            store_results(
                dataset_name=dataset_name,
                count=count, 
                definition=definition,
                query_arguments=query_arguments,
                attrs=descriptor,
                results=results
            )
    finally:
        try:
            if drop_after_test:
                algo.drop()
                print(f"清理 {definition.algorithm}")
        except Exception as e:
            print(f"清理 {definition.algorithm} 时出错: {e}")


def load_all_algorithms():
    """Load all algorithm configurations from config.yml files."""
    algorithms = []
    for config_file in glob.glob("algorithms/*/config.yml"):
        with open(config_file) as f:
            try:
                config = yaml.safe_load(f)
                if config and "float" in config and "any" in config["float"]:
                    for algo in config["float"]["any"]:
                        if not algo.get("disabled", False):
                            algorithms.append(algo)
            except Exception as e:
                print(f"加载 {config_file} 时出错: {e}")
    return algorithms


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run ANN benchmarks')
    parser.add_argument('--dataset', 
                       choices=DATASETS.keys(),
                       required=True,
                       help='数据集名称')
    parser.add_argument('--count',
                       type=int,
                       default=10,
                       help='返回的最近邻居数量')
    parser.add_argument('--runs',
                    type=int,
                    default=1,
                    help='运行测试的次数, 默认1次，因为有些数据库有缓存优化，多次运行结果会不一样。当runs>1时，只保留最后一次的运行结果')
    parser.add_argument('--duration',
                       type=int,
                       default=30,
                       help='运行测试的时间')
    parser.add_argument('--parallel_workers',
                       type=int,
                       default=8,
                       help='并行构建索引的线程数')
    parser.add_argument("--algorithms", 
                       metavar="NAME",
                       help="只运行指定的算法，多个算法用英文逗号分隔",
                       default=None)
    parser.add_argument("--copy_once", 
                       action='store_true',
                       help="每种算法类型只复制一次数据，而不是每次运行都复制",
                       default=False)
    parser.add_argument("--drop_after_test", 
                       action='store_true',
                       help="运行测试后删除表",
                       default=False)
    parser.add_argument('--reads',
                       type=int,
                       default=1,
                       help='请求的并发数')

    return parser.parse_args()


def main():
    """Main entry point to run all enabled algorithms."""
    args = parse_args()
    
    print(f"\n处理数据集: {args.dataset}")
    
    # 获取数据集信息
    dataset, dimension = get_dataset(args.dataset)
    distance = dataset.attrs["distance"]
    point_type = dataset.attrs.get("point_type", "float")

    # 使用 get_definitions 获取算法定义
    definitions = get_definitions(
        dimension=dimension,
        point_type=point_type,
        distance_metric=distance,
        count=args.count
    )
    print(f"发现 {len(definitions)} 个算法定义")

    # 如果指定了算法名称，只运行指定算法
    if args.algorithms:
        print(f"运行指定算法: {args.algorithms}")
        # 获取所有可用的算法名称
        available_algorithms = {d.algorithm for d in definitions}
        # 获取用户指定的算法名称
        specified_algorithms = set(args.algorithms.split(","))
        # 检查是否有不存在的算法
        invalid_algorithms = specified_algorithms - available_algorithms
        if invalid_algorithms:
            print(f"错误: 以下算法不存在: {', '.join(invalid_algorithms)}")
            print(f"可用算法: {', '.join(available_algorithms)}")
            return
        # 过滤出指定的算法
        definitions = [d for d in definitions if d.algorithm in specified_algorithms]


    # 记录已经复制过数据的算法
    copied_algorithms = set()

    for i, definition in enumerate(definitions):
        if args.copy_once:
            # 如果该算法类型之前没有复制过数据，则复制
            run_copy = definition.algorithm not in copied_algorithms
            if run_copy:
                copied_algorithms.add(definition.algorithm)
        else:
            run_copy = True
        print(f"\n运行 {i+1} 个算法: {definition.algorithm}, 参数: {definition.arguments}, 查询参数: {definition.query_argument_groups}")
        try:
            run(definition, 
                dataset_name=args.dataset,
                count=args.count,
                runs=args.runs,
                parallel_workers=args.parallel_workers,
                run_copy=run_copy,
                reads=args.reads,
                duration=args.duration,
                drop_after_test=args.drop_after_test)
        except Exception as e:
            print(f"运行 {definition.algorithm} 时出错: {e}")


if __name__ == "__main__":
    main()
