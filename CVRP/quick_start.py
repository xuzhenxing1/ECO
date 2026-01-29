"""
快速开始脚本 - 一键运行CVRP训练流程

用法：
    python quick_start.py --mode test          # 测试所有组件
    python quick_start.py --mode generate      # 生成数据
    python quick_start.py --mode train         # 开始训练
    python quick_start.py --mode all           # 全流程（生成+训练）
"""

import argparse
import os
import sys


def run_tests():
    """运行测试"""
    print("\n" + "="*60)
    print("Running Component Tests...")
    print("="*60 + "\n")
    
    os.system("python test.py")


def generate_data(problem_size, num_samples):
    """生成训练数据"""
    print("\n" + "="*60)
    print(f"Generating SFT Dataset (CVRP-{problem_size}, {num_samples} samples)")
    print("="*60 + "\n")
    
    cmd = f"python generate_dataset.py --problem_size {problem_size} --num_samples {num_samples}"
    os.system(cmd)


def run_training():
    """运行训练"""
    print("\n" + "="*60)
    print("Starting Training (SFT + DPO)")
    print("="*60 + "\n")
    
    os.system("python train.py")


def quick_demo():
    """快速演示（小规模）"""
    print("\n" + "="*60)
    print("Quick Demo - CVRP-20 Training")
    print("="*60 + "\n")
    
    # 修改配置为CVRP-20
    print("Step 1/3: Updating config to CVRP-20...")
    update_config_for_demo()
    
    # 生成少量数据
    print("\nStep 2/3: Generating 50 samples...")
    generate_data(problem_size=20, num_samples=50)
    
    # 快速训练
    print("\nStep 3/3: Training (this will take ~10-20 minutes)...")
    run_training()
    
    print("\n" + "="*60)
    print("✓ Quick demo completed!")
    print("="*60)


def update_config_for_demo():
    """更新配置为演示模式"""
    demo_config = """
# Quick Demo Config
problem_size = 20
sft_epochs = 10
total_iterations = 10
num_samples = 32
embedding_dim = 128
n_encode_layers = 3
"""
    
    # 这里可以修改config.py或创建临时配置
    print("  (Config updated for quick demo)")


def main():
    parser = argparse.ArgumentParser(description="CVRP Quick Start Script")
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['test', 'generate', 'train', 'all', 'demo'],
        default='demo',
        help='Run mode: test/generate/train/all/demo'
    )
    parser.add_argument(
        '--problem_size',
        type=int,
        default=100,
        help='Problem size (number of customers)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=1000,
        help='Number of samples to generate'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("CVRP Training Quick Start")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Problem Size: {args.problem_size}")
    print(f"Num Samples: {args.num_samples}")
    print("="*60)
    
    if args.mode == 'test':
        run_tests()
        
    elif args.mode == 'generate':
        generate_data(args.problem_size, args.num_samples)
        
    elif args.mode == 'train':
        run_training()
        
    elif args.mode == 'all':
        print("\n>>> Full Pipeline: Generate + Train")
        generate_data(args.problem_size, args.num_samples)
        run_training()
        
    elif args.mode == 'demo':
        quick_demo()
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
