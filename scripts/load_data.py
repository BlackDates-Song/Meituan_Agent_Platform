"""
数据加载脚本
解析 Meituan TRD 数据集中的 TSV 文件并导入 SQLite 数据库
"""

import os
import sqlite3
import pandas as pd
from pathlib import Path
import json
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MeituanDataLoader:
    """美团数据加载器"""
    
    def __init__(self, data_dir: str, db_path: str):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据集目录路径
            db_path: SQLite 数据库路径
        """
        self.data_dir = Path(data_dir)
        self.db_path = Path(db_path)
        self.conn = None
        
    def connect_db(self):
        """连接到 SQLite 数据库"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        logger.info(f"已连接到数据库: {self.db_path}")
        
    def close_db(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            logger.info("数据库连接已关闭")
            
    def create_tables(self):
        """创建数据库表"""
        cursor = self.conn.cursor()
        
        # 用户表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                avg_pay_amt TEXT,
                avg_pay_amt_weekdays TEXT,
                avg_pay_amt_weekends TEXT
            )
        """)
        
        # 商家表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pois (
                wm_poi_id INTEGER PRIMARY KEY,
                wm_poi_name TEXT,
                primary_second_tag_name TEXT,
                primary_third_tag_name TEXT,
                primary_first_tag_name TEXT,
                poi_brand_id INTEGER,
                aor_id INTEGER,
                poi_score REAL,
                delivery_comment_avg_score REAL,
                food_comment_avg_score REAL
            )
        """)
        
        # 菜品表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS spus (
                wm_food_spu_id INTEGER PRIMARY KEY,
                wm_food_spu_name TEXT,
                price REAL,
                category TEXT,
                ingredients TEXT,
                taste TEXT,
                stand_food_id INTEGER,
                stand_food_name TEXT
            )
        """)
        
        # 订单-餐厅表（训练集）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders_train (
                user_id INTEGER,
                wm_order_id INTEGER PRIMARY KEY,
                wm_poi_id INTEGER,
                aor_id INTEGER,
                order_price_interval TEXT,
                order_timestamp INTEGER,
                ord_period_name INTEGER,
                order_scene_name TEXT,
                aoi_id INTEGER,
                takedlvr_aoi_type_name TEXT,
                dt TEXT,
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                FOREIGN KEY (wm_poi_id) REFERENCES pois(wm_poi_id)
            )
        """)
        
        # 订单-菜品表（训练集）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders_spu_train (
                wm_order_id INTEGER,
                wm_food_spu_id INTEGER,
                dt TEXT,
                FOREIGN KEY (wm_order_id) REFERENCES orders_train(wm_order_id),
                FOREIGN KEY (wm_food_spu_id) REFERENCES spus(wm_food_spu_id)
            )
        """)
        
        # 用户点击序列表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders_poi_session (
                wm_order_id INTEGER,
                clicks TEXT,
                dt TEXT,
                FOREIGN KEY (wm_order_id) REFERENCES orders_train(wm_order_id)
            )
        """)
        
        # 测试集-餐厅特征表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders_test_poi (
                user_id INTEGER,
                wm_order_id INTEGER PRIMARY KEY,
                aor_id INTEGER,
                order_timestamp INTEGER,
                ord_period_name INTEGER,
                aoi_id INTEGER,
                takedlvr_aoi_type_name TEXT,
                dt TEXT,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
        
        # 测试集-菜品特征表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders_test_spu (
                user_id INTEGER,
                wm_order_id INTEGER,
                aor_id INTEGER,
                order_timestamp INTEGER,
                ord_period_name INTEGER,
                aoi_id INTEGER,
                takedlvr_aoi_type_name TEXT,
                dt TEXT,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
        
        # 测试集标签-餐厅
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders_poi_test_label (
                user_id INTEGER,
                wm_order_id INTEGER,
                wm_poi_id INTEGER,
                dt TEXT,
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                FOREIGN KEY (wm_order_id) REFERENCES orders_test_poi(wm_order_id),
                FOREIGN KEY (wm_poi_id) REFERENCES pois(wm_poi_id)
            )
        """)
        
        # 测试集标签-菜品
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders_spu_test_label (
                user_id INTEGER,
                wm_order_id INTEGER,
                wm_food_spu_id INTEGER,
                dt TEXT,
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                FOREIGN KEY (wm_order_id) REFERENCES orders_test_spu(wm_order_id),
                FOREIGN KEY (wm_food_spu_id) REFERENCES spus(wm_food_spu_id)
            )
        """)
        
        self.conn.commit()
        logger.info("数据库表创建完成")
        
    def create_indexes(self):
        """创建索引以提升查询性能"""
        cursor = self.conn.cursor()
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_orders_train_user ON orders_train(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_orders_train_poi ON orders_train(wm_poi_id)",
            "CREATE INDEX IF NOT EXISTS idx_orders_train_dt ON orders_train(dt)",
            "CREATE INDEX IF NOT EXISTS idx_orders_spu_train_order ON orders_spu_train(wm_order_id)",
            "CREATE INDEX IF NOT EXISTS idx_orders_spu_train_spu ON orders_spu_train(wm_food_spu_id)",
            "CREATE INDEX IF NOT EXISTS idx_pois_aor ON pois(aor_id)",
            "CREATE INDEX IF NOT EXISTS idx_pois_brand ON pois(poi_brand_id)",
        ]
        
        for idx_sql in indexes:
            cursor.execute(idx_sql)
            
        self.conn.commit()
        logger.info("索引创建完成")
        
    def load_tsv_file(self, filename: str, table_name: str, 
                     chunksize: int = 10000) -> int:
        """
        加载 TSV 文件到数据库
        
        Args:
            filename: 文件名
            table_name: 目标表名
            chunksize: 每批次处理的行数
            
        Returns:
            加载的总行数
        """
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            logger.warning(f"文件不存在: {file_path}")
            return 0
            
        logger.info(f"开始加载 {filename} -> {table_name}")
        
        total_rows = 0
        try:
            # 分块读取大文件
            for chunk in pd.read_csv(file_path, sep='\t', chunksize=chunksize, 
                                    na_values=['NULL', 'null', ''],
                                    keep_default_na=True):
                # 处理 NULL 值
                chunk = chunk.where(pd.notnull(chunk), None)
                
                # 写入数据库
                chunk.to_sql(table_name, self.conn, if_exists='append', 
                           index=False)
                total_rows += len(chunk)
                
                if total_rows % 50000 == 0:
                    logger.info(f"已加载 {total_rows} 行...")
                    
            logger.info(f"✓ {filename} 加载完成: {total_rows} 行")
            return total_rows
            
        except Exception as e:
            logger.error(f"加载 {filename} 失败: {str(e)}")
            raise
            
    def load_all_data(self):
        """加载所有数据文件"""
        file_mappings = [
            ('users.txt', 'users'),
            ('pois.txt', 'pois'),
            ('spus.txt', 'spus'),
            ('orders_train.txt', 'orders_train'),
            ('orders_spu_train.txt', 'orders_spu_train'),
            ('orders_poi_session.txt', 'orders_poi_session'),
            ('orders_test_poi.txt', 'orders_test_poi'),
            ('orders_test_spu.txt', 'orders_test_spu'),
            ('orders_poi_test_label.txt', 'orders_poi_test_label'),
            ('orders_spu_test_label.txt', 'orders_spu_test_label'),
        ]
        
        stats = {}
        for filename, table_name in file_mappings:
            try:
                row_count = self.load_tsv_file(filename, table_name)
                stats[table_name] = row_count
            except Exception as e:
                logger.error(f"加载表 {table_name} 时出错: {str(e)}")
                stats[table_name] = 0
                
        return stats
        
    def get_statistics(self):
        """获取数据库统计信息"""
        cursor = self.conn.cursor()
        
        tables = ['users', 'pois', 'spus', 'orders_train', 'orders_spu_train',
                 'orders_poi_session', 'orders_test_poi', 'orders_test_spu',
                 'orders_poi_test_label', 'orders_spu_test_label']
        
        stats = {}
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            stats[table] = count
            
        return stats
        
    def run(self):
        """运行完整的数据加载流程"""
        try:
            logger.info("=" * 60)
            logger.info("开始数据加载流程")
            logger.info("=" * 60)
            
            # 连接数据库
            self.connect_db()
            
            # 创建表
            logger.info("\n步骤 1: 创建数据库表...")
            self.create_tables()
            
            # 加载数据
            logger.info("\n步骤 2: 加载数据文件...")
            load_stats = self.load_all_data()
            
            # 创建索引
            logger.info("\n步骤 3: 创建索引...")
            self.create_indexes()
            
            # 输出统计信息
            logger.info("\n" + "=" * 60)
            logger.info("数据加载完成！统计信息：")
            logger.info("=" * 60)
            
            stats = self.get_statistics()
            for table, count in stats.items():
                logger.info(f"  {table:25s}: {count:>10,} 行")
                
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"数据加载过程出错: {str(e)}")
            raise
        finally:
            self.close_db()


def main():
    """主函数"""
    # 项目根目录
    project_root = Path(__file__).parent.parent
    
    # 数据目录和数据库路径
    data_dir = project_root / "Meituan_TRD"
    db_path = project_root / "data" / "db" / "meituan.db"
    
    # 创建加载器并运行
    loader = MeituanDataLoader(data_dir, db_path)
    loader.run()


if __name__ == "__main__":
    main()
