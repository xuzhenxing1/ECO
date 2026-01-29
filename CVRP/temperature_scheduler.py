# -*- coding: utf-8 -*-
"""
温度调度器 - 动态调整采样温度
与TSP版本相同，适用于CVRP的DPO训练
"""

import math


class TemperatureScheduler:
    """温度调度器，用于DPO训练中的探索/利用平衡"""
    
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
        """获取当前迭代的温度"""
        if iteration >= self.total_iterations:
            return self.end_temp
        
        progress = iteration / self.total_iterations
        
        if self.decay_type == 'linear':
            temp = self.start_temp + (self.end_temp - self.start_temp) * progress
            
        elif self.decay_type == 'exponential':
            temp = self.start_temp * (self.end_temp / self.start_temp) ** progress
            
        elif self.decay_type == 'cosine':
            temp = self.end_temp + (self.start_temp - self.end_temp) * \
                   (1 + math.cos(math.pi * progress)) / 2
                   
        elif self.decay_type == 'step':
            step = int(progress * 4)
            temp = self.start_temp - (self.start_temp - self.end_temp) * (step / 4)
            
        else:
            raise ValueError(f"Unknown decay_type: {self.decay_type}")
        
        return temp
    
    def get_info(self, iteration):
        """获取温度调度信息"""
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
            return 'Exploration'
        elif progress < 0.67:
            return 'Balance'
        else:
            return 'Exploitation'


def create_temperature_scheduler(config):
    """根据配置创建温度调度器"""
    if getattr(config, 'use_temperature_annealing', False):
        return TemperatureScheduler(
            start_temp=getattr(config, 'temperature_start', 1.5),
            end_temp=getattr(config, 'temperature_end', 0.8),
            total_iterations=config.total_iterations,
            decay_type=getattr(config, 'temperature_decay', 'linear')
        )
    else:
        # 固定温度
        class FixedTemperatureScheduler:
            def __init__(self, temp):
                self.temp = temp
            
            def get_temperature(self, iteration):
                return self.temp
            
            def get_info(self, iteration):
                return {'temperature': self.temp, 'progress': 0, 'phase': 'Fixed'}
        
        return FixedTemperatureScheduler(getattr(config, 'sampling_temperature', 1.0))
