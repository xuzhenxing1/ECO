"""
快速测试脚本 - 演示如何使用新的数据生成和训练流程
"""

import os
import sys

def print_step(step_num, title):
    print(f"\n{'='*60}")
    print(f"步骤 {step_num}: {title}")
    print(f"{'='*60}\n")

def main():
    print("\n" + "="*60)
    print("TSP-DPO-Mamba 数据生成和训练流程演示")
    print("="*60)
    
    # 步骤1: 生成SFT数据
    print_step(1, "生成SFT训练数据")
    print("命令: python generate_sft_data.py --tsp_size 20 --num_samples 1000")
    print("\n是否执行？ (y/n): ", end="")
    
    if input().lower() == 'y':
        os.system("python generate_sft_data.py --tsp_size 20 --num_samples 1000")
    else:
        print("跳过步骤1")
    
    # 步骤2: 查看生成的数据文件
    print_step(2, "查看生成的数据文件")
    data_dir = "data"
    if os.path.exists(data_dir):
        files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        if files:
            print(f"找到 {len(files)} 个数据文件:\n")
            for i, f in enumerate(files, 1):
                filepath = os.path.join(data_dir, f)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  {i}. {f} ({size_mb:.2f} MB)")
            
            latest_file = os.path.join(data_dir, sorted(files)[-1])
            print(f"\n最新文件: {latest_file}")
            
            # 步骤3: 配置训练
            print_step(3, "配置训练参数")
            print(f"需要在 config.py 中设置:")
            print(f"  sft_data_path = '{latest_file}'")
            print("\n是否自动更新config.py？ (y/n): ", end="")
            
            if input().lower() == 'y':
                # 读取config.py
                with open('config.py', 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 更新sft_data_path
                if 'sft_data_path = None' in content:
                    content = content.replace('sft_data_path = None', f'sft_data_path = "{latest_file}"')
                    with open('config.py', 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"✓ 已更新 config.py")
                else:
                    print("⚠ config.py 中已有sft_data_path配置，请手动检查")
            
            # 步骤4: 运行训练
            print_step(4, "运行训练")
            print("命令: python train.py")
            print("\n是否开始训练？ (y/n): ", end="")
            
            if input().lower() == 'y':
                os.system("python train.py")
            else:
                print("训练已跳过")
                print("\n稍后可以手动运行:")
                print("  python train.py")
        else:
            print("data/ 目录中没有数据文件")
            print("请先运行: python generate_sft_data.py")
    else:
        print("data/ 目录不存在")
        print("请先运行: python generate_sft_data.py")
    
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)
    print("\n完整工作流:")
    print("  1. python generate_sft_data.py --tsp_size 100 --num_samples 10000")
    print("  2. 在 config.py 中设置 sft_data_path")
    print("  3. python train.py")
    print("  4. python test.py")
    print("\n详细文档请查看: README_DATA.md\n")

if __name__ == "__main__":
    main()
