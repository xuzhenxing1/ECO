"""
Quick test script for LEHD data generation
Run this to verify LEHD setup is working correctly
"""

import os
import sys

# Test imports
print("Testing imports...")
try:
    import torch
    print("  ✓ PyTorch imported")
except ImportError as e:
    print(f"  ✗ PyTorch import failed: {e}")
    sys.exit(1)

# Test LEHD module
print("\nTesting LEHD module...")
LEHD_DIR = os.path.join(os.path.dirname(__file__), 'LEHD')
sys.path.insert(0, LEHD_DIR)

try:
    from LEHD.TSPModel import TSPModel
    print("  ✓ LEHD.TSPModel imported")
except ImportError as e:
    print(f"  ✗ LEHD.TSPModel import failed: {e}")
    sys.exit(1)

# Test checkpoint
print("\nChecking LEHD checkpoint...")
checkpoint_path = "LEHD/result/20230509_153705_train/checkpoint-150.pt"
if os.path.exists(checkpoint_path):
    print(f"  ✓ Checkpoint found: {checkpoint_path}")
    
    # Try loading
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"  ✓ Checkpoint loaded successfully")
        print(f"    Keys: {list(checkpoint.keys())}")
        
        # Check if model params are stored
        if 'model_params' in checkpoint:
            print(f"    Model params in checkpoint: {checkpoint['model_params']}")
    except Exception as e:
        print(f"  ✗ Checkpoint load failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print(f"  ✗ Checkpoint not found: {checkpoint_path}")
    print("\n  Available checkpoints:")
    result_dir = "LEHD/result"
    if os.path.exists(result_dir):
        for root, dirs, files in os.walk(result_dir):
            for file in files:
                if file.endswith('.pt'):
                    print(f"    - {os.path.join(root, file)}")
    sys.exit(1)

# Test data generation (small scale)
print("\nTesting LEHD data generation (10 samples)...")
try:
    from generate_sft_data_lehd import generate_sft_data_lehd
    
    filepath = generate_sft_data_lehd(
        tsp_size=20,
        num_samples=10,
        batch_size=10,
        checkpoint_path=checkpoint_path,
        save_dir="data/test"
    )
    
    if filepath and os.path.exists(filepath):
        print(f"\n  ✓ Test data generation SUCCESS!")
        print(f"    File: {filepath}")
        
        # Load and verify
        data = torch.load(filepath)
        print(f"    Problems shape: {data['problems'].shape}")
        print(f"    Tours shape: {data['tours'].shape}")
        print(f"    Avg length: {data['avg_length']:.4f}")
        
        # Cleanup test file
        os.remove(filepath)
        print(f"    Test file cleaned up")
    else:
        print(f"  ✗ Test data generation failed")
        sys.exit(1)
        
except Exception as e:
    print(f"  ✗ Error during test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✓ All tests PASSED!")
print("="*60)
print("\nYou can now run:")
print("  python generate_sft_data_lehd.py --tsp_size 50 --num_samples 5000")
print("="*60)
