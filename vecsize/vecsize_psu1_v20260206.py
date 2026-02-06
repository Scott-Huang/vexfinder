from enum import Enum, auto

# types
class QuantizerType(Enum):
    Plain = auto()
    PQ = auto()
    RabitQ = auto()
class DataType(Enum):
    Float = auto()
    Half = auto()
    Int8 = auto()
class Arch(Enum):
    ARM = auto()
    X86 = auto()

# inputs
arch = Arch.X86  # CPU架构
dim = 1024  # 向量维度
N = 100_000_000  # 数据量
m = 32  # 建索引的参数m
pq_type = QuantizerType.PQ  # 索引量化类型
data_type = DataType.Int8   # 向量类型

# optional
avg_other_data_size = 4   # 每行中其他字段占用总和的均值
concurrent_request = 300  # 峰值并发请求量



def bytes_format(s):
    bytes_format.units = ['bytes', 'KB', 'MB', 'GB', 'TB']
    for u in bytes_format.units:
        if s < 1024:
            break
        s /= 1024
    return f'{s:.3f} {u}'
def units2size(n, s):
    units2size.units = {'kb':1024, 'mb':1024 ** 2, 'gb':1024 ** 3, 'tb':1024 ** 4}
    return n * units2size.units[s.lower()]

if __name__ == '__main__':
    if arch == Arch.ARM:
        arch_alignment = 8
    elif arch == Arch.X86:
        arch_alignment = 16

    if data_type == DataType.Float:
        data_size = 4
    elif data_type == DataType.Half:
        data_size = 2
    elif data_type == DataType.Int8:
        data_size = 1

    if pq_type == QuantizerType.Plain:
        one_vec_size = data_size * dim
    elif pq_type == QuantizerType.PQ:
        if dim % 4 == 0:
            my_dim = dim / 4
        elif dim % 3 == 0:
            my_dim = dim / 3
        elif dim % 5 == 0:
            my_dim = dim / 5
        elif dim % 2 == 0:
            my_dim = dim / 2
        else:
            my_dim = dim
        one_vec_size = my_dim
    elif pq_type == QuantizerType.RabitQ:
        one_vec_size = dim * 1.5  # rough estimation

    vector_size = one_vec_size * N
    align_one_vec_size = (one_vec_size + arch_alignment - 1) // arch_alignment * arch_alignment

    avg_level = 1 / (m - 1)

    graph_size = (80 + (avg_level * 16 + 32) * m) * N
    index_size = vector_size + graph_size + units2size(8, 'KB') # 索引大小估计
    table_size = (data_size * dim * 1.28 + avg_other_data_size) * N # 表大小估计

    vector_cache_size = align_one_vec_size * N
    vector_cache_size *= 1.03
    maintenance_work_mem = align_one_vec_size * N + (128 + (avg_level * 16 + 32) * m + (avg_level + 1) * 8) * N
    maintenance_work_mem *= 1.05
    shared_cache_size = (data_size * dim * N) * 1.28 * 0.025 + avg_other_data_size * N * 0.75
    shared_cache_size *= 1.25
    shared_cache_size += graph_size * 1.02
    other_mem_size = units2size(10, 'GB')
    other_mem_size += (units2size(64, 'MB') + units2size(3, 'MB')) * concurrent_request * 1.2

    if pq_type != QuantizerType.Plain:
        index_size += units2size(8 * 6, 'KB')
        other_mem_size += units2size(8 * 8, 'KB')

    print(f'磁盘占用: {bytes_format(index_size + table_size)}')
    print(f'    表大小：{bytes_format(table_size)}')
    print(f'    索引大小：{bytes_format(index_size)}')
    print(f'        索引大小构成: 索引向量数据{bytes_format(vector_size)}, 索引图数据{bytes_format(graph_size)}\n')

    print(f'内存参数（默认全部数据为热数据的设置，对于分区冷热数据场景建议分开具体计算）: ')
    print(f'    向量缓存vector_buffers: {bytes_format(vector_cache_size)}')
    print(f'    向量索引内存构建空间maintenance_work_mem: {bytes_format(maintenance_work_mem)}')
    print(f'    通用缓存shared_buffers: {bytes_format(shared_cache_size)}')
    print(f'    预留内存空间: {bytes_format(other_mem_size)}\n')

    print(f'    不使用内存构建索引的总内存要求: {bytes_format(vector_cache_size + shared_cache_size + other_mem_size)}')
    print(f'    使用内存构建索引的总内存要求: {bytes_format(max(maintenance_work_mem, vector_cache_size) + shared_cache_size + other_mem_size)}')
    print(f'        使用内存构建索引需要设置maintenance_work_mem: {bytes_format(maintenance_work_mem)}，不使用内存构建则不需要设置该参数，建议通过会话设置参数而非直接更改全局设置')
    print(f'        Note:只构建索引不运行其他业务则可以只设置maintenance_work_mem. 大幅度缩小vector_buffers和shared_buffers参数。待索引构建完成后修改参数重启数据库。')
