"""
初始化CVRP项目目录结构
"""

import os


def create_directories():
    """创建必要的目录"""
    directories = [
        "data",           # 数据目录
        "result",         # 训练结果目录
        "configs",        # 配置目录
    ]
    
    print("Creating CVRP project directories...")
    
    for dir_name in directories:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"  ✓ Created: {dir_name}/")
        else:
            print(f"  ○ Exists: {dir_name}/")
    
    print("\n✓ Directory structure initialized!")


if __name__ == "__main__":
    create_directories()
