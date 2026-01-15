"""
数据预处理脚本
处理数据清洗、特征工程等任务
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MeituanDataPreprocessor:
    """美团数据预处理器"""
    
    def __init__(self, db_path: str):
        """
        初始化预处理器
        
        Args:
            db_path: SQLite 数据库路径
        """
        self.db_path = Path(db_path)
        self.conn = None
        
    def connect_db(self):
        """连接到数据库"""
        self.conn = sqlite3.connect(self.db_path)
        logger.info(f"已连接到数据库: {self.db_path}")
        
    def close_db(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            logger.info("数据库连接已关闭")
            
    def clean_unknown_values(self):
        """
        清理未知值（中文字符"未知"）
        将其统一替换为 NULL
        """
        logger.info("开始清理未知值...")
        
        cursor = self.conn.cursor()
        
        # 在 orders_train 表中清理
        cursor.execute("""
            UPDATE orders_train 
            SET order_scene_name = NULL 
            WHERE order_scene_name LIKE '%未知%' OR order_scene_name LIKE '%鏈煡%'
        """)
        
        cursor.execute("""
            UPDATE orders_train 
            SET takedlvr_aoi_type_name = NULL 
            WHERE takedlvr_aoi_type_name LIKE '%未知%' OR takedlvr_aoi_type_name LIKE '%鏈煡%'
        """)
        
        # 在测试集中清理
        cursor.execute("""
            UPDATE orders_test_poi 
            SET takedlvr_aoi_type_name = NULL 
            WHERE takedlvr_aoi_type_name LIKE '%未知%' OR takedlvr_aoi_type_name LIKE '%鏈煡%'
        """)
        
        cursor.execute("""
            UPDATE orders_test_spu 
            SET takedlvr_aoi_type_name = NULL 
            WHERE takedlvr_aoi_type_name LIKE '%未知%' OR takedlvr_aoi_type_name LIKE '%鏈煡%'
        """)
        
        self.conn.commit()
        logger.info("✓ 未知值清理完成")
        
    def create_user_features(self):
        """
        创建用户特征表
        包括：订单数、平均消费、活跃时段等
        """
        logger.info("创建用户特征表...")
        
        cursor = self.conn.cursor()
        
        # 创建用户统计特征表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_features AS
            SELECT 
                u.user_id,
                u.avg_pay_amt,
                u.avg_pay_amt_weekdays,
                u.avg_pay_amt_weekends,
                COUNT(o.wm_order_id) as order_count,
                COUNT(DISTINCT o.wm_poi_id) as poi_count,
                COUNT(DISTINCT o.aor_id) as aor_count,
                AVG(CASE WHEN o.ord_period_name = 0 THEN 1 ELSE 0 END) as breakfast_ratio,
                AVG(CASE WHEN o.ord_period_name = 1 THEN 1 ELSE 0 END) as lunch_ratio,
                AVG(CASE WHEN o.ord_period_name = 2 THEN 1 ELSE 0 END) as dinner_ratio,
                AVG(CASE WHEN o.ord_period_name = 3 THEN 1 ELSE 0 END) as midnight_ratio,
                MIN(o.dt) as first_order_date,
                MAX(o.dt) as last_order_date
            FROM users u
            LEFT JOIN orders_train o ON u.user_id = o.user_id
            GROUP BY u.user_id
        """)
        
        self.conn.commit()
        logger.info("✓ 用户特征表创建完成")
        
    def create_poi_features(self):
        """
        创建商家特征表
        包括：订单数、评分、热门时段等
        """
        logger.info("创建商家特征表...")
        
        cursor = self.conn.cursor()
        
        # 创建商家统计特征表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS poi_features AS
            SELECT 
                p.wm_poi_id,
                p.wm_poi_name,
                p.primary_first_tag_name,
                p.primary_second_tag_name,
                p.primary_third_tag_name,
                p.poi_brand_id,
                p.aor_id,
                p.poi_score,
                p.delivery_comment_avg_score,
                p.food_comment_avg_score,
                COUNT(o.wm_order_id) as order_count,
                COUNT(DISTINCT o.user_id) as customer_count,
                AVG(CASE WHEN o.ord_period_name = 0 THEN 1 ELSE 0 END) as breakfast_ratio,
                AVG(CASE WHEN o.ord_period_name = 1 THEN 1 ELSE 0 END) as lunch_ratio,
                AVG(CASE WHEN o.ord_period_name = 2 THEN 1 ELSE 0 END) as dinner_ratio,
                AVG(CASE WHEN o.ord_period_name = 3 THEN 1 ELSE 0 END) as midnight_ratio
            FROM pois p
            LEFT JOIN orders_train o ON p.wm_poi_id = o.wm_poi_id
            GROUP BY p.wm_poi_id
        """)
        
        self.conn.commit()
        logger.info("✓ 商家特征表创建完成")
        
    def create_spu_features(self):
        """
        创建菜品特征表
        包括：销量、价格等级等
        """
        logger.info("创建菜品特征表...")
        
        cursor = self.conn.cursor()
        
        # 创建菜品统计特征表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS spu_features AS
            SELECT 
                s.wm_food_spu_id,
                s.wm_food_spu_name,
                s.price,
                s.category,
                s.ingredients,
                s.taste,
                s.stand_food_id,
                s.stand_food_name,
                COUNT(os.wm_order_id) as order_count,
                CASE 
                    WHEN s.price < 15 THEN 'low'
                    WHEN s.price < 30 THEN 'medium'
                    WHEN s.price < 50 THEN 'high'
                    ELSE 'premium'
                END as price_level
            FROM spus s
            LEFT JOIN orders_spu_train os ON s.wm_food_spu_id = os.wm_food_spu_id
            GROUP BY s.wm_food_spu_id
        """)
        
        self.conn.commit()
        logger.info("✓ 菜品特征表创建完成")
        
    def create_interaction_features(self):
        """
        创建用户-商家/菜品交互特征
        """
        logger.info("创建交互特征表...")
        
        cursor = self.conn.cursor()
        
        # 用户-商家交互次数
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_poi_interactions AS
            SELECT 
                user_id,
                wm_poi_id,
                COUNT(*) as interaction_count,
                MAX(dt) as last_interaction_date
            FROM orders_train
            GROUP BY user_id, wm_poi_id
        """)
        
        # 用户-菜品交互次数
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_spu_interactions AS
            SELECT 
                o.user_id,
                os.wm_food_spu_id,
                COUNT(*) as interaction_count,
                MAX(os.dt) as last_interaction_date
            FROM orders_spu_train os
            JOIN orders_train o ON os.wm_order_id = o.wm_order_id
            GROUP BY o.user_id, os.wm_food_spu_id
        """)
        
        self.conn.commit()
        logger.info("✓ 交互特征表创建完成")
        
    def extract_to_csv(self, output_dir: Path):
        """
        将处理后的特征导出为 CSV 文件
        
        Args:
            output_dir: 输出目录
        """
        logger.info("导出特征到 CSV 文件...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tables = [
            'user_features',
            'poi_features', 
            'spu_features',
            'user_poi_interactions',
            'user_spu_interactions'
        ]
        
        for table in tables:
            try:
                df = pd.read_sql_query(f"SELECT * FROM {table}", self.conn)
                csv_path = output_dir / f"{table}.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"  ✓ {table}.csv ({len(df)} 行)")
            except Exception as e:
                logger.warning(f"  ✗ {table} 导出失败: {str(e)}")
                
        logger.info("CSV 导出完成")
        
    def get_data_summary(self):
        """获取数据摘要统计"""
        logger.info("\n" + "=" * 60)
        logger.info("数据摘要统计")
        logger.info("=" * 60)
        
        # 用户统计
        df = pd.read_sql_query("""
            SELECT 
                COUNT(*) as total_users,
                AVG(order_count) as avg_orders_per_user,
                MAX(order_count) as max_orders_per_user
            FROM user_features
        """, self.conn)
        logger.info(f"\n用户统计:")
        logger.info(f"  总用户数: {df['total_users'][0]:,}")
        logger.info(f"  平均订单数: {df['avg_orders_per_user'][0]:.2f}")
        logger.info(f"  最大订单数: {df['max_orders_per_user'][0]:,}")
        
        # 商家统计
        df = pd.read_sql_query("""
            SELECT 
                COUNT(*) as total_pois,
                AVG(order_count) as avg_orders_per_poi,
                AVG(poi_score) as avg_score
            FROM poi_features
        """, self.conn)
        logger.info(f"\n商家统计:")
        logger.info(f"  总商家数: {df['total_pois'][0]:,}")
        logger.info(f"  平均订单数: {df['avg_orders_per_poi'][0]:.2f}")
        logger.info(f"  平均评分: {df['avg_score'][0]:.2f}")
        
        # 菜品统计
        df = pd.read_sql_query("""
            SELECT 
                COUNT(*) as total_spus,
                AVG(price) as avg_price,
                AVG(order_count) as avg_orders_per_spu
            FROM spu_features
        """, self.conn)
        logger.info(f"\n菜品统计:")
        logger.info(f"  总菜品数: {df['total_spus'][0]:,}")
        logger.info(f"  平均价格: ¥{df['avg_price'][0]:.2f}")
        logger.info(f"  平均销量: {df['avg_orders_per_spu'][0]:.2f}")
        
        # 订单统计
        df = pd.read_sql_query("""
            SELECT 
                COUNT(*) as total_orders,
                SUM(CASE WHEN ord_period_name = 0 THEN 1 ELSE 0 END) as breakfast,
                SUM(CASE WHEN ord_period_name = 1 THEN 1 ELSE 0 END) as lunch,
                SUM(CASE WHEN ord_period_name = 2 THEN 1 ELSE 0 END) as dinner,
                SUM(CASE WHEN ord_period_name = 3 THEN 1 ELSE 0 END) as midnight
            FROM orders_train
        """, self.conn)
        logger.info(f"\n订单时段分布:")
        total = df['total_orders'][0]
        logger.info(f"  早餐: {df['breakfast'][0]:,} ({df['breakfast'][0]/total*100:.1f}%)")
        logger.info(f"  午餐: {df['lunch'][0]:,} ({df['lunch'][0]/total*100:.1f}%)")
        logger.info(f"  晚餐: {df['dinner'][0]:,} ({df['dinner'][0]/total*100:.1f}%)")
        logger.info(f"  夜宵: {df['midnight'][0]:,} ({df['midnight'][0]/total*100:.1f}%)")
        
        logger.info("=" * 60)
        
    def run(self):
        """运行完整的预处理流程"""
        try:
            logger.info("=" * 60)
            logger.info("开始数据预处理流程")
            logger.info("=" * 60)
            
            # 连接数据库
            self.connect_db()
            
            # 清理未知值
            logger.info("\n步骤 1: 清理未知值...")
            self.clean_unknown_values()
            
            # 创建特征表
            logger.info("\n步骤 2: 创建特征表...")
            self.create_user_features()
            self.create_poi_features()
            self.create_spu_features()
            self.create_interaction_features()
            
            # 导出 CSV
            logger.info("\n步骤 3: 导出特征文件...")
            project_root = Path(self.db_path).parent.parent.parent
            output_dir = project_root / "data" / "processed"
            self.extract_to_csv(output_dir)
            
            # 显示统计信息
            logger.info("\n步骤 4: 生成统计报告...")
            self.get_data_summary()
            
            logger.info("\n预处理完成！")
            
        except Exception as e:
            logger.error(f"预处理过程出错: {str(e)}")
            raise
        finally:
            self.close_db()


def main():
    """主函数"""
    # 项目根目录
    project_root = Path(__file__).parent.parent
    db_path = project_root / "data" / "db" / "meituan.db"
    
    # 检查数据库是否存在
    if not db_path.exists():
        logger.error(f"数据库不存在: {db_path}")
        logger.error("请先运行 load_data.py 加载数据")
        return
        
    # 创建预处理器并运行
    preprocessor = MeituanDataPreprocessor(db_path)
    preprocessor.run()


if __name__ == "__main__":
    main()
