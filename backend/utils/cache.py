"""
性能优化模块 - 缓存与批处理
实现LLM响应缓存、数据库查询缓存
"""

import hashlib
import json
import time
from typing import Any, Optional, Callable
from functools import wraps
import pickle
from pathlib import Path


class SimpleCache:
    """简单的内存缓存实现"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        初始化缓存
        
        Args:
            max_size: 最大缓存条目数
            ttl: 缓存过期时间（秒）
        """
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
        self.access_count = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def _make_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        key_data = {
            'args': args,
            'kwargs': kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            
            # 检查是否过期
            if time.time() - timestamp < self.ttl:
                self.hit_count += 1
                self.access_count[key] = self.access_count.get(key, 0) + 1
                return value
            else:
                # 过期，删除
                del self.cache[key]
        
        self.miss_count += 1
        return None
    
    def set(self, key: str, value: Any):
        """设置缓存值"""
        # 如果缓存已满，删除最少使用的条目
        if len(self.cache) >= self.max_size:
            # LFU策略：删除访问次数最少的
            if self.access_count:
                least_used_key = min(self.access_count, key=self.access_count.get)
                del self.cache[least_used_key]
                del self.access_count[least_used_key]
            else:
                # 如果没有访问记录，删除最早的
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
        
        self.cache[key] = (value, time.time())
        self.access_count[key] = 0
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_count.clear()
        self.hit_count = 0
        self.miss_count = 0
    
    def get_stats(self) -> dict:
        """获取缓存统计信息"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": f"{hit_rate:.2%}",
            "total_requests": total_requests
        }


class PersistentCache:
    """持久化缓存（基于文件）"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{key}.pkl"
    
    def get(self, key: str, max_age: int = 86400) -> Optional[Any]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            max_age: 最大缓存时间（秒）
        """
        cache_file = self._get_cache_path(key)
        
        if cache_file.exists():
            # 检查文件年龄
            file_age = time.time() - cache_file.stat().st_mtime
            
            if file_age < max_age:
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    print(f"Cache load error: {e}")
                    return None
        
        return None
    
    def set(self, key: str, value: Any):
        """设置缓存值"""
        cache_file = self._get_cache_path(key)
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            print(f"Cache save error: {e}")
    
    def clear(self):
        """清空所有缓存"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()


# 全局缓存实例
llm_cache = SimpleCache(max_size=500, ttl=3600)  # LLM响应缓存1小时
query_cache = SimpleCache(max_size=1000, ttl=300)  # 查询缓存5分钟
vector_cache = PersistentCache("data/cache/vectors")  # 向量缓存


def cache_llm_response(func: Callable) -> Callable:
    """LLM响应缓存装饰器"""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 生成缓存键
        cache_key = llm_cache._make_key(*args, **kwargs)
        
        # 尝试从缓存获取
        cached_result = llm_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # 缓存未命中，调用原函数
        result = func(*args, **kwargs)
        
        # 存入缓存
        if result is not None:
            llm_cache.set(cache_key, result)
        
        return result
    
    return wrapper


def cache_query_result(ttl: int = 300):
    """数据库查询缓存装饰器"""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = query_cache._make_key(func.__name__, *args, **kwargs)
            
            # 尝试从缓存获取
            cached_result = query_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 缓存未命中，调用原函数
            result = func(*args, **kwargs)
            
            # 存入缓存
            if result is not None:
                query_cache.set(cache_key, result)
            
            return result
        
        return wrapper
    
    return decorator


class BatchProcessor:
    """批处理器 - 用于批量处理请求"""
    
    def __init__(self, batch_size: int = 10, max_wait_time: float = 0.5):
        """
        初始化批处理器
        
        Args:
            batch_size: 批次大小
            max_wait_time: 最大等待时间（秒）
        """
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_items = []
        self.last_batch_time = time.time()
    
    def add(self, item: Any) -> bool:
        """
        添加项到批处理队列
        
        Returns:
            是否触发批处理
        """
        self.pending_items.append(item)
        
        # 检查是否应该触发批处理
        should_process = (
            len(self.pending_items) >= self.batch_size or
            time.time() - self.last_batch_time >= self.max_wait_time
        )
        
        return should_process
    
    def get_batch(self) -> list:
        """获取当前批次并清空队列"""
        batch = self.pending_items.copy()
        self.pending_items.clear()
        self.last_batch_time = time.time()
        return batch


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = {}
    
    def record(self, operation: str, duration: float):
        """记录操作耗时"""
        if operation not in self.metrics:
            self.metrics[operation] = {
                "count": 0,
                "total_time": 0,
                "min_time": float('inf'),
                "max_time": 0
            }
        
        metric = self.metrics[operation]
        metric["count"] += 1
        metric["total_time"] += duration
        metric["min_time"] = min(metric["min_time"], duration)
        metric["max_time"] = max(metric["max_time"], duration)
    
    def get_stats(self, operation: str = None) -> dict:
        """获取性能统计"""
        if operation:
            if operation in self.metrics:
                metric = self.metrics[operation]
                return {
                    "operation": operation,
                    "count": metric["count"],
                    "avg_time": metric["total_time"] / metric["count"],
                    "min_time": metric["min_time"],
                    "max_time": metric["max_time"],
                    "total_time": metric["total_time"]
                }
            return {}
        
        # 返回所有统计
        stats = {}
        for op, metric in self.metrics.items():
            stats[op] = {
                "count": metric["count"],
                "avg_time": metric["total_time"] / metric["count"],
                "min_time": metric["min_time"],
                "max_time": metric["max_time"]
            }
        return stats
    
    def reset(self):
        """重置统计"""
        self.metrics.clear()


# 全局性能监控器
perf_monitor = PerformanceMonitor()


def monitor_performance(operation_name: str = None):
    """性能监控装饰器"""
    
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                perf_monitor.record(op_name, duration)
        
        return wrapper
    
    return decorator


if __name__ == "__main__":
    # 测试缓存
    print("="*60)
    print("缓存测试")
    print("="*60)
    
    @cache_llm_response
    def mock_llm_call(prompt: str) -> str:
        print(f"  调用LLM: {prompt}")
        time.sleep(0.1)  # 模拟API延迟
        return f"Response to: {prompt}"
    
    # 第一次调用
    print("\n第一次调用:")
    result1 = mock_llm_call("测试问题1")
    print(f"  结果: {result1}")
    
    # 第二次调用（应该命中缓存）
    print("\n第二次调用（相同问题）:")
    result2 = mock_llm_call("测试问题1")
    print(f"  结果: {result2}")
    
    # 缓存统计
    print("\n缓存统计:")
    stats = llm_cache.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 性能监控测试
    print("\n" + "="*60)
    print("性能监控测试")
    print("="*60)
    
    @monitor_performance("数据库查询")
    def mock_db_query():
        time.sleep(0.05)
        return [{"id": 1, "name": "测试"}]
    
    # 执行多次查询
    for i in range(5):
        mock_db_query()
    
    # 显示性能统计
    print("\n性能统计:")
    stats = perf_monitor.get_stats()
    for op, metrics in stats.items():
        print(f"\n{op}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}s" if 'time' in key else f"  {key}: {value}")
