# 数据库连接管理重构

## 概述

为了改进数据库连接的管理，我们将数据库连接相关的逻辑从`OpenGaussAdapter`类移到了`engine.py`中。这样做的好处是：

1. 集中管理数据库连接，使代码更加清晰
2. 每个操作都使用新的连接，避免连接被长时间占用
3. 操作完成后立即释放连接，减少资源占用
4. 提高并发性能，避免连接冲突

## 主要更改

### 1. 创建数据库引擎

在`vecindex_finder/core/engine.py`中，我们创建了一个`DatabaseEngine`类，负责管理数据库连接：

```python
class DatabaseEngine:
    """数据库引擎，负责管理数据库连接"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化数据库引擎"""
        self.config = None
        self.load_config(config_path)
    
    def get_connection(self):
        """获取数据库连接"""
        # 创建并返回新的数据库连接
        
    def close_connection(self, connection, cursor=None):
        """关闭数据库连接"""
        # 关闭连接和游标
```

### 2. 修改适配器类

在`OpenGaussAdapter`类中，我们修改了以下方法，使其使用新的连接管理方式：

- `get_connection`: 使用`db_engine`获取连接
- `connect`: 使用`db_engine`获取连接
- `close`: 使用`db_engine`关闭连接
- `create_index`: 为每个操作创建新的连接
- `test_performance`: 为测试性能创建新的连接
- `compute_recall`: 为计算召回率创建新的连接
- `get_groundtruth_results`: 为获取真实结果创建新的连接
- `drop_index`: 为删除索引创建新的连接

### 3. 并发处理

在`test_performance`方法中，我们为每个工作线程创建单独的连接，避免连接冲突：

```python
def worker(query_id, query_vector):
    # 为每个工作线程创建新的连接
    worker_connection = self.get_connection()
    worker_cursor = worker_connection.cursor()
    
    try:
        # 执行查询
        # ...
    finally:
        # 关闭连接
        worker_cursor.close()
        worker_connection.close()
```

## 测试

我们创建了一个测试文件`cursor/test_db_connection.py`，用于测试新的数据库连接管理功能：

```python
def test_db_engine():
    """测试数据库引擎"""
    # 测试获取连接和执行查询
    
def test_adapter():
    """测试适配器"""
    # 测试适配器的连接管理
```

## 注意事项

1. 每个操作都会创建新的连接，这可能会增加连接开销，但可以避免连接被长时间占用
2. 操作完成后立即释放连接，减少资源占用
3. 并发操作时，每个线程使用单独的连接，避免连接冲突
4. 配置文件仍然从适配器目录加载，但连接管理逻辑已移到`engine.py`中 