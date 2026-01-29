"""
LKH求解器测试脚本
用于验证LKH安装和配置是否正确

Usage:
    python test_lkh.py
"""

import sys
import platform

def test_lkh_installation():
    """测试LKH是否正确安装和配置"""
    print("\n" + "="*60)
    print("LKH求解器测试")
    print("="*60)
    print(f"系统: {platform.system()}")
    print(f"Python版本: {sys.version.split()[0]}")
    print("="*60 + "\n")
    
    # 1. 测试导入
    print("步骤1: 测试模块导入...")
    try:
        from lkh_solver import LKHSolver
        from config import Config
        print("✓ 模块导入成功\n")
    except ImportError as e:
        print(f"✗ 模块导入失败: {e}\n")
        return False
    
    # 2. 测试LKH路径
    print("步骤2: 检查LKH配置...")
    print(f"Config中的LKH路径: {Config.lkh_path}")
    
    # 3. 初始化求解器
    print("\n步骤3: 初始化LKH求解器...")
    try:
        solver = LKHSolver()
        print("✓ LKH求解器初始化成功\n")
    except FileNotFoundError as e:
        print(f"✗ LKH求解器初始化失败:")
        print(f"  {e}\n")
        print("解决方案:")
        if platform.system() == "Windows":
            print("  1. 确保已下载并解压LKH-Windows版本")
            print("  2. 检查config.py中的lkh_path是否指向LKH-3.exe（注意是LKH-3.exe不是LKH.exe）")
            print(f"     当前路径: {Config.lkh_path}")
            print("  3. 默认路径: D:\\lkh-w\\LKHWin-3.0.13\\x64\\Release\\LKH-3.exe")
            print("  4. 或使用: D:\\lkh-w\\LKHWin-3.0.13\\x64\\Debug\\LKH-3.exe")
        else:
            print("  1. 安装LKH: sudo apt-get install lkh (或从源码编译)")
            print("  2. 确保LKH在系统PATH中，或在config.py中指定完整路径")
        return False
    
    # 4. 测试求解小规模问题
    print("步骤4: 测试求解TSP-10问题...")
    try:
        import numpy as np
        import torch
        from tsp_env import TSPEnv
        
        # 生成小规模测试问题
        np.random.seed(42)
        coords = np.random.rand(10, 2)
        
        print(f"  生成随机TSP-10问题...")
        print(f"  坐标形状: {coords.shape}")
        
        # 求解
        print(f"  调用LKH求解器...")
        tour = solver.solve(coords, verbose=False)
        
        print(f"  ✓ 求解成功!")
        print(f"  解的长度: {len(tour)}")
        print(f"  访问顺序: {tour}")
        
        # 验证解的有效性
        assert len(tour) == 10, "路径长度不正确"
        assert len(set(tour)) == 10, "路径中有重复节点"
        assert all(0 <= node < 10 for node in tour), "节点索引超出范围"
        
        # 计算路径长度
        def compute_length(coords, tour):
            length = 0
            for i in range(len(tour)):
                j = (i + 1) % len(tour)
                length += np.linalg.norm(coords[tour[i]] - coords[tour[j]])
            return length
        
        tour_length = compute_length(coords, tour)
        print(f"  路径总长度: {tour_length:.4f}\n")
        
    except Exception as e:
        print(f"✗ 求解测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. 性能测试
    print("步骤5: 性能测试（TSP-20，5个样本）...")
    try:
        import time
        
        test_size = 20
        num_tests = 5
        
        times = []
        for i in range(num_tests):
            coords = np.random.rand(test_size, 2)
            
            start = time.time()
            tour = solver.solve(coords, verbose=False)
            elapsed = time.time() - start
            
            times.append(elapsed)
            print(f"  样本 {i+1}/{num_tests}: {elapsed:.2f}秒")
        
        avg_time = np.mean(times)
        print(f"\n  平均求解时间: {avg_time:.2f}秒/样本")
        print(f"  预估TSP-100: ~{avg_time * 5:.1f}秒/样本\n")
        
    except Exception as e:
        print(f"⚠ 性能测试失败（非致命）: {e}\n")
    
    # 总结
    print("="*60)
    print("✓ LKH求解器测试通过！")
    print("="*60)
    print("\n可以开始使用LKH生成高质量数据:")
    print("  python generate_sft_data_lkh.py --tsp_size 50 --num_samples 100\n")
    
    return True


def main():
    success = test_lkh_installation()
    
    if not success:
        print("\n" + "="*60)
        print("测试失败 - 请检查LKH安装")
        print("="*60 + "\n")
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
