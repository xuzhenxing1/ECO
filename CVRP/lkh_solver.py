"""
LKH-3求解器封装 for CVRP
支持 Windows 和 Ubuntu 系统
"""

import os
import platform
import subprocess
import tempfile
import numpy as np
import torch
from pathlib import Path


class LKHSolver:
    """LKH-3求解器封装类（用于CVRP）"""
    
    def __init__(self, lkh_path=None, runs=10, max_trials=1000, seed=None):
        """
        初始化LKH求解器
        
        Args:
            lkh_path: LKH-3可执行文件的路径
            runs: LKH运行次数（越多质量越高但越慢）
            max_trials: 最大尝试次数
            seed: 随机种子
        """
        self.system = platform.system()
        self.runs = runs
        self.max_trials = max_trials
        self.seed = seed
        
        if lkh_path is None:
            if self.system == "Windows":
                lkh_path = r"D:\lkh-w\LKHWin-3.0.13\x64\Release\LKH-3.exe"
            else:
                lkh_path = "LKH"
        
        self.lkh_path = lkh_path
        self._verify_lkh()
    
    def _verify_lkh(self):
        """验证LKH是否可用"""
        try:
            if self.system == "Windows":
                if not os.path.exists(self.lkh_path):
                    raise FileNotFoundError(
                        f"LKH executable not found at: {self.lkh_path}"
                    )
            
            print(f"✓ LKH solver found: {self.lkh_path}")
            print(f"✓ System: {self.system}")
            
        except FileNotFoundError:
            raise FileNotFoundError(
                f"LKH solver not found!\n"
                f"System: {self.system}\n"
                f"Expected path: {self.lkh_path}\n"
                f"Please install LKH-3 or specify the correct path."
            )
    
    def write_vrp_file(self, depot_coord, node_coords, node_demands, capacity, filepath):
        """
        将CVRP问题写入VRP格式文件（TSPLIB格式）
        
        Args:
            depot_coord: [2] depot坐标
            node_coords: [N, 2] 客户节点坐标
            node_demands: [N] 客户需求量
            capacity: 车辆容量
            filepath: 保存路径
        """
        n = len(node_coords)
        
        # 坐标缩放到整数
        depot_scaled = (depot_coord * 10000).astype(int)
        coords_scaled = (node_coords * 10000).astype(int)
        demands_scaled = (node_demands * 10000).astype(int)
        capacity_scaled = int(capacity * 10000)
        
        with open(filepath, 'w') as f:
            f.write(f"NAME : cvrp{n}\n")
            f.write(f"COMMENT : Generated CVRP\n")
            f.write(f"TYPE : CVRP\n")
            f.write(f"DIMENSION : {n+1}\n")  # depot + customers
            f.write(f"EDGE_WEIGHT_TYPE : EUC_2D\n")
            f.write(f"CAPACITY : {capacity_scaled}\n")
            f.write(f"NODE_COORD_SECTION\n")
            
            # Depot (索引1)
            f.write(f"1 {depot_scaled[0]} {depot_scaled[1]}\n")
            
            # Customers (索引2~N+1)
            for i, (x, y) in enumerate(coords_scaled, 2):
                f.write(f"{i} {x} {y}\n")
            
            f.write(f"DEMAND_SECTION\n")
            
            # Depot需求=0
            f.write(f"1 0\n")
            
            # Customers需求
            for i, demand in enumerate(demands_scaled, 2):
                f.write(f"{i} {demand}\n")
            
            f.write(f"DEPOT_SECTION\n")
            f.write(f"1\n")  # Depot索引
            f.write(f"-1\n")
            f.write(f"EOF\n")
    
    def write_parameter_file(self, vrp_filepath, tour_filepath, param_filepath):
        """
        创建LKH参数文件
        
        Args:
            vrp_filepath: VRP问题文件路径
            tour_filepath: 解输出文件路径
            param_filepath: 参数文件保存路径
        """
        with open(param_filepath, 'w') as f:
            f.write(f"PROBLEM_FILE = {vrp_filepath}\n")
            f.write(f"OUTPUT_TOUR_FILE = {tour_filepath}\n")
            f.write(f"RUNS = {self.runs}\n")
            f.write(f"MAX_TRIALS = {self.max_trials}\n")
            
            if self.seed is not None:
                f.write(f"SEED = {self.seed}\n")
            
            # CVRP特定参数
            f.write(f"MOVE_TYPE = 5\n")
            f.write(f"PATCHING_C = 3\n")
            f.write(f"PATCHING_A = 2\n")
    
    def parse_tour_file(self, tour_filepath, dimension):
        """
        解析LKH输出的tour文件
        
        Args:
            tour_filepath: tour文件路径
            dimension: 问题规模（depot + customers）
            
        Returns:
            tour: list of int, 访问顺序（0-indexed）
                 depot=0, customers=1~N
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
                        # LKH使用1-indexed，转为0-indexed
                        tour.append(node - 1)
                    except ValueError:
                        continue
        
        return tour
    
    def solve(self, depot_coord, node_coords, node_demands, capacity=1.0, verbose=False):
        """
        使用LKH求解单个CVRP问题
        
        Args:
            depot_coord: numpy array [2]
            node_coords: numpy array [N, 2]
            node_demands: numpy array [N]
            capacity: 车辆容量（归一化值）
            verbose: 是否打印详细信息
            
        Returns:
            tour: list of int, 访问顺序（0-indexed）
        """
        # 转换为numpy
        if isinstance(depot_coord, torch.Tensor):
            depot_coord = depot_coord.cpu().numpy()
        if isinstance(node_coords, torch.Tensor):
            node_coords = node_coords.cpu().numpy()
        if isinstance(node_demands, torch.Tensor):
            node_demands = node_demands.cpu().numpy()
        
        n_customers = len(node_coords)
        dimension = n_customers + 1  # depot + customers
        
        # 创建临时文件
        with tempfile.TemporaryDirectory() as tmpdir:
            vrp_file = os.path.join(tmpdir, "problem.vrp")
            param_file = os.path.join(tmpdir, "params.par")
            tour_file = os.path.join(tmpdir, "solution.tour")
            
            # 写入VRP文件
            self.write_vrp_file(depot_coord, node_coords, node_demands, capacity, vrp_file)
            
            # 写入参数文件
            self.write_parameter_file(vrp_file, tour_file, param_file)
            
            # 调用LKH
            try:
                result = subprocess.run(
                    [self.lkh_path, param_file],
                    capture_output=True,
                    text=True,
                    stdin=subprocess.DEVNULL,
                    timeout=300
                )
                
                if verbose:
                    print("LKH stdout:", result.stdout)
                    if result.stderr:
                        print("LKH stderr:", result.stderr)
                
                if result.returncode != 0 and result.returncode != 1:
                    raise RuntimeError(
                        f"LKH failed with return code {result.returncode}\n"
                        f"stderr: {result.stderr}"
                    )
                
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    f"LKH timeout after 300 seconds for CVRP-{n_customers}"
                )
            
            # 解析结果
            if not os.path.exists(tour_file):
                raise RuntimeError(
                    f"LKH did not generate tour file: {tour_file}"
                )
            
            tour = self.parse_tour_file(tour_file, dimension)
        
        return tour
    
    def solve_batch(self, depot_coords, node_coords_batch, node_demands_batch, 
                   capacity=1.0, verbose=False):
        """
        批量求解CVRP问题
        
        Args:
            depot_coords: [B, 2]
            node_coords_batch: [B, N, 2]
            node_demands_batch: [B, N]
            capacity: 车辆容量
            verbose: 是否打印详细信息
            
        Returns:
            tours: list of list, 每个问题的解
        """
        if isinstance(depot_coords, torch.Tensor):
            depot_coords = depot_coords.cpu().numpy()
        if isinstance(node_coords_batch, torch.Tensor):
            node_coords_batch = node_coords_batch.cpu().numpy()
        if isinstance(node_demands_batch, torch.Tensor):
            node_demands_batch = node_demands_batch.cpu().numpy()
        
        tours = []
        batch_size = len(depot_coords)
        
        for i in range(batch_size):
            if verbose or (i + 1) % 10 == 0:
                print(f"  Solving {i+1}/{batch_size}...", end='\r')
            
            tour = self.solve(
                depot_coords[i], 
                node_coords_batch[i], 
                node_demands_batch[i],
                capacity,
                verbose=False
            )
            tours.append(tour)
        
        if verbose or batch_size >= 10:
            print(f"  Solved {batch_size}/{batch_size} problems.")
        
        return tours


def test_lkh_solver():
    """测试LKH求解器"""
    print("\n" + "="*60)
    print("Testing LKH-3 Solver for CVRP")
    print("="*60 + "\n")
    
    # 创建测试问题
    np.random.seed(42)
    depot = np.random.rand(2)
    nodes = np.random.rand(10, 2)
    demands = np.random.randint(1, 10, 10) / 30.0
    capacity = 1.0
    
    print(f"Test problem: CVRP-{len(nodes)}")
    print(f"Depot: {depot}")
    print(f"Demands: {demands}")
    print(f"Capacity: {capacity}\n")
    
    # 初始化求解器
    solver = LKHSolver()
    
    # 求解
    print("Solving with LKH-3...")
    tour = solver.solve(depot, nodes, demands, capacity, verbose=True)
    
    print(f"\nSolution tour: {tour}")
    print(f"Tour length: {len(tour)}")
    
    print("\n✓ LKH Solver test passed!\n")


if __name__ == "__main__":
    test_lkh_solver()
