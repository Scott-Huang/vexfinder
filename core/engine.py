import psycopg
from core.config import Config, config
from core.logging import logger
from typing import Optional

class DatabaseEngine:
    def __init__(self, config_obj: Optional[Config] = None):
        self.config = config_obj or config

    def get_connection(self):
        try:
            conn = psycopg.connect(
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
