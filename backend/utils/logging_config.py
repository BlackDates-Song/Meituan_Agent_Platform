"""
日志和监控配置
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict
import json
import time
from functools import wraps


# 创建日志目录
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


# 日志格式配置
class JSONFormatter(logging.Formatter):
    """JSON格式日志"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # 添加异常信息
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # 添加额外字段
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data
        
        return json.dumps(log_data, ensure_ascii=False)


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    配置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别
    
    Returns:
        配置好的logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 控制台输出 (文本格式)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # 文件输出 (JSON格式)
    log_file = LOG_DIR / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)
    
    # 错误日志单独记录
    error_file = LOG_DIR / f"{name}_error_{datetime.now().strftime('%Y%m%d')}.log"
    error_handler = logging.FileHandler(error_file, encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter())
    logger.addHandler(error_handler)
    
    return logger


# 全局logger
api_logger = setup_logger("api")
service_logger = setup_logger("service")
model_logger = setup_logger("model")


def log_execution_time(logger: logging.Logger = None):
    """
    记录函数执行时间的装饰器
    
    Args:
        logger: 日志记录器，默认使用api_logger
    """
    if logger is None:
        logger = api_logger
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            func_name = f"{func.__module__}.{func.__name__}"
            
            try:
                logger.info(f"开始执行: {func_name}")
                result = func(*args, **kwargs)
                
                elapsed = time.time() - start_time
                logger.info(
                    f"执行完成: {func_name}",
                    extra={"extra_data": {"execution_time": f"{elapsed:.3f}s"}}
                )
                
                return result
            
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"执行失败: {func_name}",
                    exc_info=True,
                    extra={"extra_data": {"execution_time": f"{elapsed:.3f}s"}}
                )
                raise
        
        return wrapper
    return decorator


class RequestLogger:
    """API请求日志中间件"""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or api_logger
    
    def __call__(self, request, call_next):
        """FastAPI中间件"""
        import asyncio
        
        async def log_request():
            start_time = time.time()
            request_id = id(request)
            
            # 记录请求
            self.logger.info(
                f"请求开始: {request.method} {request.url.path}",
                extra={"extra_data": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "client": request.client.host if request.client else None
                }}
            )
            
            # 处理请求
            try:
                response = await call_next(request)
                
                elapsed = time.time() - start_time
                self.logger.info(
                    f"请求完成: {request.method} {request.url.path}",
                    extra={"extra_data": {
                        "request_id": request_id,
                        "status_code": response.status_code,
                        "execution_time": f"{elapsed:.3f}s"
                    }}
                )
                
                return response
            
            except Exception as e:
                elapsed = time.time() - start_time
                self.logger.error(
                    f"请求失败: {request.method} {request.url.path}",
                    exc_info=True,
                    extra={"extra_data": {
                        "request_id": request_id,
                        "execution_time": f"{elapsed:.3f}s"
                    }}
                )
                raise
        
        return asyncio.create_task(log_request())


class MetricsCollector:
    """性能指标收集器"""
    
    def __init__(self):
        self.metrics: Dict[str, list] = {}
        self.logger = setup_logger("metrics")
    
    def record(self, metric_name: str, value: float, tags: Dict[str, Any] = None):
        """
        记录指标
        
        Args:
            metric_name: 指标名称
            value: 指标值
            tags: 标签
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append({
            "timestamp": datetime.now().isoformat(),
            "value": value,
            "tags": tags or {}
        })
        
        # 日志记录
        self.logger.debug(
            f"指标记录: {metric_name}",
            extra={"extra_data": {"value": value, "tags": tags}}
        )
    
    def get_stats(self, metric_name: str) -> Dict[str, Any]:
        """获取指标统计"""
        if metric_name not in self.metrics:
            return {}
        
        values = [m["value"] for m in self.metrics[metric_name]]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "recent": values[-10:]  # 最近10个值
        }
    
    def export_all(self) -> Dict[str, Any]:
        """导出所有指标"""
        return {
            metric_name: self.get_stats(metric_name)
            for metric_name in self.metrics.keys()
        }
    
    def clear(self, metric_name: str = None):
        """清除指标数据"""
        if metric_name:
            self.metrics.pop(metric_name, None)
        else:
            self.metrics.clear()


# 全局指标收集器
metrics_collector = MetricsCollector()


def track_metric(metric_name: str, tags: Dict[str, Any] = None):
    """
    跟踪函数执行时间的装饰器
    
    Args:
        metric_name: 指标名称
        tags: 标签
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                
                # 记录成功执行时间
                metrics_collector.record(
                    f"{metric_name}.success",
                    elapsed,
                    tags
                )
                
                return result
            
            except Exception as e:
                elapsed = time.time() - start_time
                
                # 记录失败执行时间
                metrics_collector.record(
                    f"{metric_name}.error",
                    elapsed,
                    {**(tags or {}), "error": str(e)}
                )
                
                raise
        
        return wrapper
    return decorator


# 使用示例
if __name__ == "__main__":
    # 日志示例
    logger = setup_logger("test")
    
    logger.info("普通信息日志")
    logger.warning("警告日志")
    logger.error("错误日志", extra={"extra_data": {"user_id": 123}})
    
    try:
        raise ValueError("测试异常")
    except Exception:
        logger.exception("异常日志")
    
    # 装饰器示例
    @log_execution_time(logger)
    @track_metric("test_function")
    def test_function():
        time.sleep(0.5)
        return "完成"
    
    test_function()
    
    # 查看指标
    print("\n指标统计:")
    print(json.dumps(metrics_collector.export_all(), indent=2, ensure_ascii=False))
