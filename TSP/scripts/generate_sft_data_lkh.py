"""
Generate high-quality SFT training data using LKH solver
Supports both Windows and Ubuntu systems

LKH (Lin-Kernighan-Helsgaun) is one of the best TSP heuristic solvers
capable of generating near-optimal solutions

Usage:
    # Windows
    python generate_sft_data_lkh.py --tsp_size 100 --num_samples 1000
    
    # Ubuntu (make sure LKH is in system PATH)
    python generate_sft_data_lkh.py --tsp_size 100 --num_samples 1000 --lkh_path lkh
"""

# ============================================================
# Configuration - Modify parameters here
# ============================================================

# Basic data generation config
DEFAULT_TSP_SIZE = None  # TSP size, None means use value from config.py
DEFAULT_NUM_SAMPLES = 1000  # Number of samples to generate
DEFAULT_BATCH_SIZE = 10  # Batch size for processing

# LKH quality parameters - KEY CONFIG
LKH_RUNS = 10  # LKH runs (1=fast, 10=standard, 20-50=high quality)
LKH_MAX_TRIALS = 1000  # Max trials (1000=standard, 5000=high quality)


# Other config
SAVE_DIR = "data"  # Directory to save data
LKH_PATH = None  # LKH path, None means auto-detect

# ============================================================
# Quality presets (uncomment to use)
# ============================================================
# Fast test (lower quality, faster)
# LKH_RUNS = 1
# LKH_MAX_TRIALS = 500

# Standard quality (recommended)
# LKH_RUNS = 10
# LKH_MAX_TRIALS = 1000

# High quality (for formal experiments)
# LKH_RUNS = 20
# LKH_MAX_TRIALS = 3000

# Ultimate quality (for papers, very slow)
# LKH_RUNS = 50
# LKH_MAX_TRIALS = 5000

# ============================================================

import torch
import argparse
import os
from datetime import datetime
from tqdm import tqdm
import platform

from config import Config
from tsp_env import TSPEnv
from lkh_solver import LKHSolver


def ensure_dir(path: str) -> str:
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)
    return path


def generate_sft_data_lkh(
    tsp_size, 
    num_samples, 
    batch_size=10,  # LKH is slower, use smaller batch
    save_dir="data",
    lkh_path=None,
    lkh_runs=10,  # Number of LKH runs
    lkh_max_trials=1000  # LKH max trials
):
    """
    Generate SFT training data using LKH solver
    
    Args:
        tsp_size: TSP problem size (number of cities)
        num_samples: Total number of samples to generate
        batch_size: Batch size for processing (LKH is slow, recommend 10-50)
        save_dir: Directory to save data
        lkh_path: Path to LKH executable
        lkh_runs: Number of LKH runs, more runs = higher quality (default 10, recommend 10-50)
        lkh_max_trials: Maximum trials (default 1000)
    
    Returns:
        Path to saved file
    """
    print(f"\n{'='*60}")
    print(f"Generate High-Quality SFT Data Using LKH Solver")
    print(f"{'='*60}")
    print(f"System: {platform.system()}")
    print(f"TSP Size: {tsp_size}")
    print(f"Num Samples: {num_samples}")
    print(f"Batch Size: {batch_size}")
    print(f"LKH Runs: {lkh_runs} (per problem)")
    print(f"LKH Max Trials: {lkh_max_trials}")
    print(f"Save Dir: {save_dir}")
    print(f"{'='*60}\n")
    
    # Initialize environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    env = TSPEnv(device)
    
    # Initialize LKH solver
    print("\nInitializing LKH solver...")
    try:
        solver = LKHSolver(lkh_path, runs=lkh_runs, max_trials=lkh_max_trials)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease check LKH installation:")
        print("  Windows: Make sure path is correct, default is D:\\lkh-w\\LKHWin-3.0.13\\LKH-3\\x64\\Release\\LKH.exe")
        print("  Ubuntu: Make sure LKH is in system PATH, or use --lkh_path to specify")
        return None
    
    # Create save directory
    ensure_dir(save_dir)
    
    # Prepare lists to store data
    all_problems = []
    all_tours = []
    
    # Generate data in batches
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"\nStarting data generation (Estimated time: TSP-{tsp_size} ~1-10 sec per sample)...\n")
    
    for batch_idx in tqdm(range(num_batches), desc="Batch Progress"):
        # Calculate actual batch size
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
        
        # 1. Generate random TSP problems
        problems = env.get_random_problems(current_batch_size, tsp_size)
        
        # 2. Generate high-quality solutions using LKH solver
        with torch.no_grad():
            # LKH solve (returns list of lists)
            tours_list = solver.solve_batch(problems, verbose=False)
            
            # CRITICAL FIX: Normalize tours to start from node 0
            # LKH returns tours starting from arbitrary nodes (e.g., [245, 12, 89, ...])
            # But model training expects tours starting from node 0
            normalized_tours = []
            for tour in tours_list:
                # Find where node 0 appears in the tour
                idx_of_zero = tour.index(0)
                # Rotate tour to start from node 0
                normalized_tour = tour[idx_of_zero:] + tour[:idx_of_zero]
                normalized_tours.append(normalized_tour)
            
            # Convert to tensor
            tours = torch.tensor(normalized_tours, dtype=torch.long, device=device)
        
        # 3. Move data to CPU and store
        all_problems.append(problems.cpu())
        all_tours.append(tours.cpu())
    
    # Concatenate all batches
    all_problems = torch.cat(all_problems, dim=0)
    all_tours = torch.cat(all_tours, dim=0)
    
    # Keep only required number of samples
    all_problems = all_problems[:num_samples]
    all_tours = all_tours[:num_samples]
    
    print(f"\nData generation completed!")
    print(f"Problem shape: {all_problems.shape}")  # [num_samples, tsp_size, 2]
    print(f"Tour shape: {all_tours.shape}")        # [num_samples, tsp_size]
    
    # Calculate statistics
    print("\nCalculating data quality statistics...")
    with torch.no_grad():
        problems_gpu = all_problems.to(device)
        tours_gpu = all_tours.to(device)
        lengths = env.get_tour_length(problems_gpu, tours_gpu)
        avg_length = lengths.mean().item()
        std_length = lengths.std().item()
        min_length = lengths.min().item()
        max_length = lengths.max().item()
    
    print(f"\nGenerated data statistics (LKH solver):")
    print(f"  Avg tour length: {avg_length:.4f}")
    print(f"  Std deviation: {std_length:.4f}")
    print(f"  Min tour length: {min_length:.4f}")
    print(f"  Max tour length: {max_length:.4f}")
    
    # Save data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sft_data_lkh_tsp{tsp_size}_n{num_samples}_{timestamp}.pt"
    filepath = os.path.join(save_dir, filename)
    
    torch.save({
        'problems': all_problems,      # [num_samples, tsp_size, 2]
        'tours': all_tours,            # [num_samples, tsp_size]
        'tsp_size': tsp_size,
        'num_samples': num_samples,
        'avg_length': avg_length,
        'std_length': std_length,
        'min_length': min_length,
        'max_length': max_length,
        'timestamp': timestamp,
        'algorithm': 'LKH',  # Mark as LKH-generated
        'solver_version': 'LKH-3'
    }, filepath)
    
    print(f"\nData saved to: {filepath}")
    print(f"{'='*60}\n")
    
    return filepath


def compare_with_nearest_neighbor(tsp_size=20, num_test=10):
    """
    Compare solution quality: LKH vs Nearest Neighbor
    """
    print(f"\n{'='*60}")
    print(f"Quality Comparison: LKH vs Nearest Neighbor")
    print(f"{'='*60}\n")
    
    from heuristics import get_nearest_neighbor_tour
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TSPEnv(device)
    solver = LKHSolver()
    
    # Generate test problems
    problems = env.get_random_problems(num_test, tsp_size)
    
    # LKH solve
    print("Solving with LKH...")
    lkh_tours = solver.solve_batch(problems)
    lkh_tours_tensor = torch.tensor(lkh_tours, dtype=torch.long, device=device)
    lkh_lengths = env.get_tour_length(problems, lkh_tours_tensor)
    
    # Nearest Neighbor solve
    print("Solving with Nearest Neighbor...")
    with torch.no_grad():
        nn_tours = get_nearest_neighbor_tour(problems)
    nn_lengths = env.get_tour_length(problems, nn_tours)
    
    # Statistics comparison
    lkh_avg = lkh_lengths.mean().item()
    nn_avg = nn_lengths.mean().item()
    improvement = ((nn_avg - lkh_avg) / nn_avg) * 100
    
    print(f"\nComparison Results (TSP-{tsp_size}, {num_test} samples):")
    print(f"  LKH avg length:      {lkh_avg:.4f}")
    print(f"  NN avg length:       {nn_avg:.4f}")
    print(f"  LKH improvement:     {improvement:.2f}%")
    print(f"\nLKH generates significantly better solutions than Nearest Neighbor!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate high-quality TSP SFT training data using LKH solver'
    )
    parser.add_argument('--tsp_size', type=int, default=DEFAULT_TSP_SIZE,
                        help=f'TSP problem size (default: {DEFAULT_TSP_SIZE or "value from config.py"})')
    parser.add_argument('--num_samples', type=int, default=DEFAULT_NUM_SAMPLES,
                        help=f'Number of samples to generate (default: {DEFAULT_NUM_SAMPLES})')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Batch size (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR,
                        help=f'Directory to save data (default: {SAVE_DIR})')
    parser.add_argument('--lkh_path', type=str, default=LKH_PATH,
                        help='Path to LKH executable (default: auto-detect)')
    parser.add_argument('--lkh_runs', type=int, default=LKH_RUNS,
                        help=f'Number of LKH runs (default: {LKH_RUNS})')
    parser.add_argument('--lkh_max_trials', type=int, default=LKH_MAX_TRIALS,
                        help=f'LKH max trials (default: {LKH_MAX_TRIALS})')
    parser.add_argument('--compare', action='store_true',
                        help='Compare LKH vs Nearest Neighbor quality')
    
    args = parser.parse_args()
    
    # If just comparing
    if args.compare:
        compare_with_nearest_neighbor()
        return
    
    # If tsp_size not specified, use value from config
    tsp_size = args.tsp_size if args.tsp_size is not None else Config.tsp_size
    
    # Generate data
    filepath = generate_sft_data_lkh(
        tsp_size=tsp_size,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        lkh_path=args.lkh_path,
        lkh_runs=args.lkh_runs,
        lkh_max_trials=args.lkh_max_trials
    )
    
    if filepath:
        print("Success! High-quality data generated!")
        print(f"Use during training: --sft_data_path {filepath}")
        print("\nTip: LKH solutions are much better than Nearest Neighbor")
        print("     Using this data for SFT will give better initial policy")


if __name__ == "__main__":
    main()
