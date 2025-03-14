import psycopg2
from core.config import Config, config
from core.logging import logger
from typing import Optional

class DatabaseEngine:
    """数据库引擎，负责管理数据库连接"""
    
    def __init__(self, config_obj: Optional[Config] = None):
        """
        初始化数据库引擎
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
        """

        self.config = config_obj or config
    
    def get_connection(self):
        """
        获取数据库连接
        
        Returns:
            psycopg2.connection: 数据库连接对象
            
        Raises:
            ValueError: 如果未提供连接配置
            ConnectionError: 如果连接失败
        """
        try:
            conn = psycopg2.connect(
                host=self.config.connection.host,
                port=self.config.connection.port,
                user=self.config.connection.user,
                password=self.config.connection.password,
                dbname=self.config.connection.dbname
            )
            conn.autocommit = True
            return conn
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise ConnectionError(f"无法连接到数据库: {e}")

    
    def close_connection(self, connection, cursor=None):
        """
        关闭数据库连接
        
        Args:
            connection: 数据库连接对象
            cursor: 数据库游标对象
        """
        if cursor:
            try:
                cursor.close()
            except Exception as e:
                logger.warning(f"关闭游标失败: {e}")
        
        if connection:
            try:
                connection.close()
                logger.debug("数据库连接已关闭")
            except Exception as e:
                logger.warning(f"关闭连接失败: {e}")


db_engine = DatabaseEngine(config)