import logging
import os
from datetime import datetime

# 创建logs目录
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# 生成带时间戳的日志文件名
log_file = os.path.join(log_dir, f"vecindex_finder_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),  # 文件处理器
        logging.StreamHandler()         # 控制台处理器
    ]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
