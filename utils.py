import logging
from logging.handlers import TimedRotatingFileHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),        # 输出到控制台
        TimedRotatingFileHandler(
            'log.log',        
            when="midnight",  # 按天分割
            interval=1,       # 每 1 天分割一次
            backupCount=7     # 保留全部的日志
        )
    ]
)
# 日志记录器
logger = logging.getLogger()
