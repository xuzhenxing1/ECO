"""
配置预设 - 不同问题规模的推荐配置
"""

class ConfigPresets:
    """预设配置"""
    
    @staticmethod
    def get_cvrp20_config():
        """CVRP-20 快速验证配置"""
        return {
            'problem_size': 20,
            'demand_scaler': 30,
            
            # 模型
            'embedding_dim': 128,
            'n_encode_layers': 3,
            
            # SFT
            'sft_lr': 3e-4,
            'sft_batch_size': 256,
            'sft_epochs': 20,
            
            # DPO
            'dpo_lr': 1e-4,
            'dpo_batch_size': 64,
            'dpo_beta': 0.3,
            'total_iterations': 20,
            'epochs_per_iter': 3,
            'num_samples': 32,
            'num_pairs_per_sample': 16,
            
            # 评估
            'eval_batch_size': 200,
        }
    
    @staticmethod
    def get_cvrp50_config():
        """CVRP-50 中等规模配置"""
        return {
            'problem_size': 50,
            'demand_scaler': 40,
            
            # 模型
            'embedding_dim': 192,
            'n_encode_layers': 4,
            
            # SFT
            'sft_lr': 4e-4,
            'sft_batch_size': 384,
            'sft_epochs': 40,
            
            # DPO
            'dpo_lr': 1e-4,
            'dpo_batch_size': 56,
            'dpo_beta': 0.3,
            'total_iterations': 50,
            'epochs_per_iter': 4,
            'num_samples': 64,
            'num_pairs_per_sample': 24,
            
            # 评估
            'eval_batch_size': 300,
        }
    
    @staticmethod
    def get_cvrp100_config():
        """CVRP-100 大规模高质量配置"""
        return {
            'problem_size': 100,
            'demand_scaler': 50,
            
            # 模型
            'embedding_dim': 256,
            'n_encode_layers': 6,
            
            # SFT
            'sft_lr': 4.5e-4,
            'sft_batch_size': 512,
            'sft_epochs': 60,
            
            # DPO
            'dpo_lr': 1e-4,
            'dpo_batch_size': 48,
            'dpo_beta': 0.3,
            'total_iterations': 100,
            'epochs_per_iter': 5,
            'num_samples': 128,
            'num_pairs_per_sample': 32,
            
            # 评估
            'eval_batch_size': 500,
        }
    
    @staticmethod
    def apply_preset(config_obj, preset_name):
        """应用预设到Config对象"""
        presets = {
            'cvrp20': ConfigPresets.get_cvrp20_config(),
            'cvrp50': ConfigPresets.get_cvrp50_config(),
            'cvrp100': ConfigPresets.get_cvrp100_config(),
        }
        
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Choose from {list(presets.keys())}")
        
        preset = presets[preset_name]
        
        for key, value in preset.items():
            setattr(config_obj, key, value)
        
        print(f"✓ Applied preset: {preset_name}")
        print(f"  Problem size: {preset['problem_size']}")
        print(f"  Embedding dim: {preset['embedding_dim']}")
        print(f"  Encode layers: {preset['n_encode_layers']}")


def main():
    """打印所有预设配置"""
    print("="*60)
    print("CVRP Configuration Presets")
    print("="*60)
    
    presets = {
        'CVRP-20': ConfigPresets.get_cvrp20_config(),
        'CVRP-50': ConfigPresets.get_cvrp50_config(),
        'CVRP-100': ConfigPresets.get_cvrp100_config(),
    }
    
    for name, config in presets.items():
        print(f"\n{name}:")
        print("-" * 40)
        for key, value in config.items():
            print(f"  {key:25s}: {value}")
    
    print("\n" + "="*60)
    print("Usage:")
    print("  from configs.config_presets import ConfigPresets")
    print("  ConfigPresets.apply_preset(Config, 'cvrp100')")
    print("="*60)


if __name__ == "__main__":
    main()
