import pandas as pd
import os

# ⚠️ 请确保路径是你本地的实际路径
DATA_DIR = "../Meituan_TRD" 

files_to_check = [
    "users.txt", 
    "pois.txt", 
    "spus.txt", 
    "orders_train.txt"
]

def explore_file(file_name):
    file_path = os.path.join(DATA_DIR, file_name)
    print(f"--- 正在分析: {file_name} ---")
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return

    try:
        # ✅ 修正点：使用 sep='\t' 处理制表符
        df = pd.read_csv(file_path, sep='\t', nrows=5) 
        
        print(f"✅ 读取成功! 列名列表:")
        print(df.columns.tolist())
        
        print(">> 前2行样本数据:")
        print(df.head(2))
        
        # 顺便看下数据类型，这对建 MySQL 表很重要
        print(">> 数据类型预览:")
        print(df.dtypes)
        print("\n" + "="*30 + "\n")
        
    except Exception as e:
        print(f"❌ 读取出错: {e}")

# 执行分析
for f in files_to_check:
    explore_file(f)