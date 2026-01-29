"""
LKH求解器封装工具类
支持 Windows 和 Ubuntu 系统
"""

import os
import sys
import platform
import subprocess
import tempfile
import numpy as np
import torch
from pathlib import Path


class LKHSolver:
    """LKH求解器封装类"""
    
    def __init__(self, lkh_path=None, runs=10, max_trials=1000, seed=None):
        """
        初始化LKH求解器
        
        Args:
            lkh_path: LKH可执行文件的路径
                     - Windows: "D:\\lkh-w\\LKHWin-3.0.13\\LKH-3\\x64\\Release\\LKH.exe"
                     - Ubuntu: "/usr/local/bin/LKH" 或 "lkh"
            runs: LKH运行次数，越多质量越高但越慢（默认10，推荐10-50）
            max_trials: 最大尝试次数，控制搜索深度（默认1000）
            seed: 随机种子，None表示使用随机值
        """
        self.system = platform.system()
        self.runs = runs
        self.max_trials = max_trials
        self.seed = seed
        
        if lkh_path is None:
            # 根据系统自动检测
            if self.system == "Windows":
                lkh_path = r"D:\lkh-w\LKHWin-3.0.13\x64\Release\LKH-3.exe"
            else:  # Linux/Ubuntu
                lkh_path = "LKH"  # 假设在系统PATH中
        
        self.lkh_path = lkh_path
        self._verify_lkh()
    
    def _verify_lkh(self):
        """验证LKH是否可用"""
        try:
            if self.system == "Windows":
                if not os.path.exists(self.lkh_path):
                    raise FileNotFoundError(
                        f"LKH executable not found at: {self.lkh_path}\n"
                        f"Please check the path or install LKH."
                    )
            else:
                # Linux: 尝试运行LKH命令
                result = subprocess.run(
                    [self.lkh_path],
                    capture_output=True,
                    timeout=5
                )
                # LKH通常会因为缺少参数而退出，这是正常的
            
            print(f"✓ LKH solver found: {self.lkh_path}")
            print(f"✓ System: {self.system}")
            
        except FileNotFoundError:
            raise FileNotFoundError(
                f"LKH solver not found!\n"
                f"System: {self.system}\n"
                f"Expected path: {self.lkh_path}\n"
                f"Please install LKH or specify the correct path."
            )
        except subprocess.TimeoutExpired:
            # 超时是正常的，说明LKH存在
            print(f"✓ LKH solver found: {self.lkh_path}")
    
    def write_tsplib_file(self, coords, filepath):
        """
        将坐标写入TSPLIB格式文件
        
        Args:
            coords: numpy array [N, 2], 坐标范围 [0, 1]
            filepath: TSPLIB文件保存路径
        """
        n = len(coords)
        
        # 将 [0,1] 坐标缩放到整数 (LKH需要整数坐标)
        # 缩放到 [0, 10000] 以保持精度
        coords_scaled = (coords * 10000).astype(int)
        
        with open(filepath, 'w') as f:
            f.write(f"NAME : tsp{n}\n")
            f.write(f"COMMENT : Generated TSP\n")
            f.write(f"TYPE : TSP\n")
            f.write(f"DIMENSION : {n}\n")
            f.write(f"EDGE_WEIGHT_TYPE : EUC_2D\n")
            f.write(f"NODE_COORD_SECTION\n")
            
            for i, (x, y) in enumerate(coords_scaled, 1):
                f.write(f"{i} {x} {y}\n")
            
            f.write("EOF\n")
    
    def write_parameter_file(self, tsp_filepath, tour_filepath, param_filepath):
        """
        创建LKH参数文件
        
        Args:
            tsp_filepath: TSP问题文件路径
            tour_filepath: 解输出文件路径
            param_filepath: 参数文件保存路径
        """
        with open(param_filepath, 'w') as f:
            f.write(f"PROBLEM_FILE = {tsp_filepath}\n")
            f.write(f"OUTPUT_TOUR_FILE = {tour_filepath}\n")
            
            # 核心质量参数
            f.write(f"RUNS = {self.runs}\n")  # 运行次数：越多越好，但越慢
            f.write(f"MAX_TRIALS = {self.max_trials}\n")  # 最大尝试次数
            
            # 随机种子
            if self.seed is not None:
                f.write(f"SEED = {self.seed}\n")
            
            # 高级参数（提高质量）
            f.write(f"MOVE_TYPE = 2\n")  # 5-opt moves（更强但更慢，默认5）
            f.write(f"PATCHING_C = 3\n")  # Patching策略
            f.write(f"PATCHING_A = 2\n")
            f.write(f"EXCESS = 0.05\n")  # 候选边的超额百分比
    
    def parse_tour_file(self, tour_filepath, dimension):
        """
        解析LKH输出的tour文件
        
        Args:
            tour_filepath: tour文件路径
            dimension: TSP问题规模
            
        Returns:
            tour: list of int, 访问顺序 (1-indexed转为0-indexed)
        """
        tour = []
        reading_tour = False
        
        with open(tour_filepath, 'r') as f:
            for line in f:
                line = line.strip()
                
                if line == "TOUR_SECTION":
                    reading_tour = True
                    continue
                
                if reading_tour:
                    if line == "-1" or line == "EOF":
                        break
                    try:
                        node = int(line)
                        tour.append(node - 1)  # 转为0-indexed
                    except ValueError:
                        continue
        
        if len(tour) != dimension:
            raise ValueError(
                f"Tour length mismatch: expected {dimension}, got {len(tour)}"
            )
        
        return tour
    
    def solve(self, coords, verbose=False):
        """
        使用LKH求解单个TSP问题
        
        Args:
            coords: numpy array [N, 2] or torch.Tensor [N, 2]
            verbose: 是否打印详细信息
            
        Returns:
            tour: list of int [N], 访问顺序 (0-indexed)
        """
        # 转换为numpy
        if isinstance(coords, torch.Tensor):
            coords = coords.cpu().numpy()
        
        dimension = len(coords)
        
        # 创建临时文件
        with tempfile.TemporaryDirectory() as tmpdir:
            tsp_file = os.path.join(tmpdir, "problem.tsp")
            param_file = os.path.join(tmpdir, "params.par")
            tour_file = os.path.join(tmpdir, "solution.tour")
            
            # 写入TSPLIB格式文件
            self.write_tsplib_file(coords, tsp_file)
            
            # 写入参数文件
            self.write_parameter_file(tsp_file, tour_file, param_file)
            
            # 调用LKH
            try:
                result = subprocess.run(
                    [self.lkh_path, param_file],
                    capture_output=True,
                    text=True,
                    stdin=subprocess.DEVNULL,  # 防止等待标准输入
                    timeout=300  # 5分钟超时
                )
                
                if verbose:
                    print("LKH stdout:", result.stdout)
                    if result.stderr:
                        print("LKH stderr:", result.stderr)
                
                if result.returncode != 0 and result.returncode != 1:
                    # LKH有时返回1但仍然成功
                    raise RuntimeError(
                        f"LKH failed with return code {result.returncode}\n"
                        f"stderr: {result.stderr}"
                    )
                
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    f"LKH timeout after 300 seconds for TSP-{dimension}"
                )
            
            # 解析结果
            if not os.path.exists(tour_file):
                raise RuntimeError(
                    f"LKH did not generate tour file: {tour_file}"
                )
            
            tour = self.parse_tour_file(tour_file, dimension)
        
        return tour
    
    def solve_batch(self, coords_batch, verbose=False):
        """
        批量求解TSP问题
        
        Args:
            coords_batch: numpy array [B, N, 2] or torch.Tensor [B, N, 2]
            verbose: 是否打印详细信息
            
        Returns:
            tours: list of list, 每个问题的解
        """
        if isinstance(coords_batch, torch.Tensor):
            coords_batch = coords_batch.cpu().numpy()
        
        tours = []
        batch_size = len(coords_batch)
        
        for i, coords in enumerate(coords_batch):
            if verbose or (i + 1) % 10 == 0:
                print(f"  Solving {i+1}/{batch_size}...", end='\r')
            
            tour = self.solve(coords, verbose=False)
            tours.append(tour)
        
        if verbose or batch_size >= 10:
            print(f"  Solved {batch_size}/{batch_size} problems.")
        
        return tours


def test_lkh_solver():
    """测试LKH求解器"""
    print("\n" + "="*60)
    print("Testing LKH Solver")
    print("="*60 + "\n")
    
    # 创建测试问题
    np.random.seed(42)
    coords = np.random.rand(20, 2)
    
    print(f"Test problem: TSP-{len(coords)}")
    print(f"Coordinates shape: {coords.shape}\n")
    
    # 初始化求解器
    solver = LKHSolver()
    
    # 求解
    print("Solving with LKH...")
    tour = solver.solve(coords, verbose=True)
    
    print(f"\nSolution tour: {tour}")
    print(f"Tour length: {len(tour)}")
    
    # 计算路径长度
    def compute_tour_length(coords, tour):
        length = 0
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            length += np.linalg.norm(coords[tour[i]] - coords[tour[j]])
        return length
    
    length = compute_tour_length(coords, tour)
    print(f"Tour cost: {length:.4f}")
    
    print("\n✓ LKH Solver test passed!\n")


if __name__ == "__main__":
    test_lkh_solver()
