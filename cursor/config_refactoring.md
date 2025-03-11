# 配置管理重构

## 概述

为了改进配置管理，我们将 OpenGaussAdapter 中的配置加载和检查逻辑拆分到 config.py 中的 OpenGaussConfig 类中。这样做的好处是：

1. 集中管理配置，使代码更加清晰
2. 提供统一的配置访问接口
3. 减少重复代码
4. 便于扩展和维护

## 主要更改

### 1. 创建 OpenGaussConfig 类

在 `vecindex_finder/core/config.py` 中，我们创建了一个 `OpenGaussConfig` 类，继承自 `Config` 类，专门用于管理 OpenGauss 数据库的配置：

```python
class OpenGaussConfig(Config):
    """OpenGauss数据库配置管理类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化OpenGauss配置管理器"""
        # 初始化配置属性
        # ...
        
        # 调用父类初始化方法
        super().__init__(config_path)
        
        # 检查并设置OpenGauss特定配置
        self.check_and_setup_config()
```

### 2. 修改 DatabaseEngine 类

在 `vecindex_finder/core/engine.py` 中，我们修改了 `DatabaseEngine` 类，使其使用 `OpenGaussConfig` 类：

```python
class DatabaseEngine:
    """数据库引擎，负责管理数据库连接"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化数据库引擎"""
        self.config_manager = OpenGaussConfig(config_path)
    
    def get_connection(self):
        """获取数据库连接"""
        connection_config = self.config_manager.get_connection_config()
        # ...
    
    def get_config_manager(self) -> OpenGaussConfig:
        """获取配置管理器"""
        return self.config_manager
```

### 3. 修改 OpenGaussAdapter 类

在 `vecindex_finder/db_adapters/opengauss/adapter.py` 中，我们修改了 `OpenGaussAdapter` 类，使其使用 `OpenGaussConfig` 类：

```python
class OpenGaussAdapter(VectorDatabaseAdapter):
    """OpenGauss数据库适配器"""
    
    def __init__(self):
        """初始化OpenGauss适配器"""
        # ...
        
        # 获取配置管理器
        self.config_manager = db_engine.get_config_manager()
        
        # 从配置管理器获取配置
        self.table_name, self.vector_column_name, self.dimension, self.metric = self.config_manager.get_table_info()
        # ...
```

## 配置访问接口

`OpenGaussConfig` 类提供了以下配置访问接口：

1. `get_connection_config()`: 获取数据库连接配置
2. `get_table_info()`: 获取表信息
3. `get_index_config()`: 获取索引配置
4. `get_performance_config()`: 获取性能配置
5. `get_query_config()`: 获取查询配置
6. `get_sampling_config()`: 获取采样配置

## 测试

我们创建了一个测试文件 `cursor/test_db_connection.py`，用于测试新的配置管理方式：

```python
def test_config():
    """测试配置管理"""
    # 创建配置管理器
    config_manager = OpenGaussConfig()
    
    # 获取配置
    connection_config = config_manager.get_connection_config()
    table_info = config_manager.get_table_info()
    # ...
```

## 注意事项

1. 配置文件仍然从 OpenGauss 适配器目录加载，但配置管理逻辑已移至 `config.py` 中
2. `OpenGaussAdapter` 类中的 `load_and_check_config` 方法已被废弃，保留是为了向后兼容
3. 所有配置访问都应通过 `OpenGaussConfig` 类提供的接口进行 