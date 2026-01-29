# -*- coding: utf-8 -*-
"""
温度调度器 - 动态调整采样温度

支持多种退火策略：
1. Linear: 线性递减
2. Exponential: 指数递减
3. Cosine: 余弦退火
4. Step: 阶梯式下降
"""

import math


class TemperatureScheduler:
    """温度调度器，用于 DPO 训练中的探索/利用平衡"""
    
    def __init__(self, 
                 start_temp=1.5, 
                 end_temp=0.8, 
                 total_iterations=100,
                 decay_type='linear'):
        """
        Args:
            start_temp: 初始温度（通常 > 1.0，鼓励探索）
            end_temp: 最终温度（通常 < 1.0，鼓励利用）
            total_iterations: 总迭代次数
            decay_type: 退火类型 ('linear', 'exponential', 'cosine', 'step')
        """
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.total_iterations = total_iterations
        self.decay_type = decay_type
        
    def get_temperature(self, iteration):
        """获取当前迭代的温度
        
        Args:
            iteration: 当前迭代次数 (0-based)
        
        Returns:
            temperature: 当前温度值
        """
        if iteration >= self.total_iterations:
            return self.end_temp
        
        progress = iteration / self.total_iterations  # 0.0 -> 1.0
        
        if self.decay_type == 'linear':
            # 线性插值
            temp = self.start_temp + (self.end_temp - self.start_temp) * progress
            
        elif self.decay_type == 'exponential':
            # 指数衰减：T = T_start * (T_end/T_start)^progress
            temp = self.start_temp * (self.end_temp / self.start_temp) ** progress
            
        elif self.decay_type == 'cosine':
            # 余弦退火（平滑下降）
            temp = self.end_temp + (self.start_temp - self.end_temp) * \
                   (1 + math.cos(math.pi * progress)) / 2
                   
        elif self.decay_type == 'step':
            # 阶梯下降（每 1/4 迭代降一次）
            step = int(progress * 4)  # 0, 1, 2, 3
            temp = self.start_temp - (self.start_temp - self.end_temp) * (step / 4)
            
        else:
            raise ValueError(f"Unknown decay_type: {self.decay_type}")
        
        return temp
    
    def get_info(self, iteration):
        """获取温度调度信息（用于日志）"""
        temp = self.get_temperature(iteration)
        progress = iteration / self.total_iterations * 100
        return {
            'temperature': temp,
            'progress': progress,
            'phase': self._get_phase(iteration)
        }
    
    def _get_phase(self, iteration):
        """判断当前处于哪个阶段"""
        progress = iteration / self.total_iterations
        if progress < 0.33:
            return 'Exploration'  # 探索阶段
        elif progress < 0.67:
            return 'Balance'      # 平衡阶段
        else:
            return 'Exploitation'  # 利用阶段


class AdaptiveTemperatureScheduler(TemperatureScheduler):
    """自适应温度调度器 - 根据训练表现动态调整"""
    
    def __init__(self, 
                 start_temp=1.5, 
                 end_temp=0.8,
                 total_iterations=100,
                 decay_type='cosine',
                 adaptation_rate=0.1):
        """
        Args:
            adaptation_rate: 自适应调整速率（0.0-1.0）
        """
        super().__init__(start_temp, end_temp, total_iterations, decay_type)
        self.adaptation_rate = adaptation_rate
        self.history = []  # 存储历史性能
        
    def adapt_temperature(self, current_temp, performance_metric):
        """根据性能指标自适应调整温度
        
        Args:
            current_temp: 当前温度
            performance_metric: 性能指标（如 winner-loser gap）
                - gap 太小 (< 0.01): 增加温度，增强探索
                - gap 适中 (0.01-0.05): 保持温度
                - gap 太大 (> 0.05): 降低温度，增强利用
        
        Returns:
            adjusted_temp: 调整后的温度
        """
        self.history.append(performance_metric)
        
        # 简单的自适应规则
        if performance_metric < 0.01:
            # Gap 太小，需要更多探索
            adjustment = 1.0 + self.adaptation_rate
        elif performance_metric > 0.05:
            # Gap 太大，需要更多利用
            adjustment = 1.0 - self.adaptation_rate
        else:
            # Gap 适中，保持当前温度
            adjustment = 1.0
        
        adjusted_temp = current_temp * adjustment
        
        # 限制在合理范围内
        adjusted_temp = max(0.5, min(2.0, adjusted_temp))
        
        return adjusted_temp


def create_temperature_scheduler(config):
    """从配置创建温度调度器"""
    use_annealing = getattr(config, 'use_temperature_annealing', False)
    
    if not use_annealing:
        # 固定温度
        class FixedTemperature:
            def __init__(self, temp):
                self.temp = temp
            
            def get_temperature(self, iteration):
                return self.temp
            
            def get_info(self, iteration):
                return {'temperature': self.temp, 'progress': 0, 'phase': 'Fixed'}
        
        return FixedTemperature(getattr(config, 'sampling_temperature', 1.0))
    
    # 使用退火策略
    return TemperatureScheduler(
        start_temp=getattr(config, 'temperature_start', 1.5),
        end_temp=getattr(config, 'temperature_end', 0.8),
        total_iterations=getattr(config, 'total_iterations', 100),
        decay_type=getattr(config, 'temperature_decay', 'linear')
    )


# ========== 使用示例 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("温度调度器演示")
    print("=" * 60)
    
    schedulers = {
        'Linear': TemperatureScheduler(1.5, 0.8, 100, 'linear'),
        'Exponential': TemperatureScheduler(1.5, 0.8, 100, 'exponential'),
        'Cosine': TemperatureScheduler(1.5, 0.8, 100, 'cosine'),
        'Step': TemperatureScheduler(1.5, 0.8, 100, 'step'),
    }
    
    print("\n不同退火策略在关键迭代点的温度值：")
    print("-" * 60)
    print(f"{'Iteration':<12} {'Linear':<12} {'Exponential':<12} {'Cosine':<12} {'Step':<12}")
    print("-" * 60)
    
    for iter_num in [0, 10, 25, 50, 75, 90, 99]:
        temps = [scheduler.get_temperature(iter_num) for scheduler in schedulers.values()]
        print(f"{iter_num:<12} {temps[0]:<12.3f} {temps[1]:<12.3f} {temps[2]:<12.3f} {temps[3]:<12.3f}")
    
    print("\n" + "=" * 60)
    print("推荐使用 'cosine' 退火策略，提供最平滑的过渡")
    print("=" * 60)
