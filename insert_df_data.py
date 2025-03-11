import pandas as pd
import psycopg2
from io import StringIO
import numpy as np
from urllib.parse import quote_plus


# 示例数据
def create_sample_df():
    data = {
        'id': range(1, 1001),
        'name': [f'name_{i}' for i in range(1, 1001)],
        'embedding': [np.random.rand(1536).tolist() for _ in range(1000)]
    }
    return pd.DataFrame(data)


# 数据库连接参数
DB_PARAMS = {
    'dbname': 'ann',
    'user': 'ann',
    'password': 'Huawei@123',
    'host': '127.0.0.1',
    'port': '6432'
}


def create_table():
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS df_table (id INT, name VARCHAR, embedding floatvector((1536)))")
    conn.commit()
    cur.close()
    conn.close()


def drop_table():
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS df_table")
    conn.commit()
    cur.close()
    conn.close()


def select_table(embedding):
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute(f"SELECT id, name, embedding <-> '{embedding}' as distance FROM df_table order by distance limit 1")
    result = cur.fetchall()
    print(result)
    cur.close()
    conn.close()


def reset_table():
    drop_table()
    create_table()

# 1. 使用 psycopg2 单条插入
def insert_one_by_one(df):
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    try:
        for index, row in df.iterrows():
            cur.execute(
                "INSERT INTO df_table (id, name, embedding) VALUES (%s, %s, %s)",
                (row['id'], row['name'], row['embedding'])
            )
        conn.commit()
        print("单条插入完成")
    except Exception as e:
        print(f"插入错误: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


# 2. 使用 psycopg2 COPY 批量插入
def copy_insert(df):
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    try:
        # 将 DataFrame 转换为 CSV 格式的字符串
        output = StringIO()
        df.to_csv(output, sep='\t', header=False, index=False)
        output.seek(0)

        # 使用 COPY 命令批量插入
        cur.copy_from(
            output,
            'df_table',
            columns=('id', 'name', 'embedding')
        )
        conn.commit()
        print("COPY 插入完成")
    except Exception as e:
        print(f"插入错误: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


# 3. 使用 SQLAlchemy 插入
def sqlalchemy_insert(df):
    try:
        from sqlalchemy import create_engine
        # 重写PostgreSQL方言类的版本检测方法,以解决SQLAlchemy与数据库版本不匹配的问题
        from sqlalchemy.dialects.postgresql.base import PGDialect
        def _get_server_version_info(self, connection):
            return (9.6, 0, 0)  # specify the version of pg redshift emulates. this is just an example
        PGDialect._get_server_version_info = _get_server_version_info
        # 创建 SQLAlchemy 引擎
        engine = create_engine(
            f'postgresql://{DB_PARAMS["user"]}:{quote_plus(DB_PARAMS["password"])}@{DB_PARAMS["host"]}:{DB_PARAMS["port"]}/{DB_PARAMS["dbname"]}'
        )

        # 使用 to_sql 方法插入数据
        df.to_sql(
            'df_table',
            engine,
            if_exists='append',
            index=False
        )
        print("SQLAlchemy 插入完成")
    except Exception as e:
        print(f"插入错误: {e}")


def main():
    # 创建示例数据
    df = create_sample_df()

    # 测试三种插入方法

    print("\n1. 测试 psycopg2 单条插入")
    reset_table()
    insert_one_by_one(df)
    select_table(df.iloc[12]['embedding'])
    reset_table()

    print("\n2. 测试 psycopg2 COPY 插入")
    copy_insert(df)
    select_table(df.iloc[12]['embedding'])
    reset_table()

    print("\n3. 测试 SQLAlchemy 插入")
    sqlalchemy_insert(df)
    select_table(df.iloc[123]['embedding'])
    drop_table()


if __name__ == "__main__":
    main()
