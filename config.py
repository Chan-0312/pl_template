from pydantic_settings import BaseSettings
from typing import Literal, Optional
import argparse
from distutils.util import strtobool


class TrainerSettings(BaseSettings):
    # 基础参数
    seed: int = 159456                      # 随机种子
    epochs: int = 50                       # 训练轮数
    batch_size: int = 32                    # 训练批次数
    accumulate_grad_batches: int = 1        # 梯度累积
    num_workers: int = 4                    # 训练数据加载线程数
    lr: float = 1e-3                        # 学习率
    model_name: str = 'your_net'            # 模型名称
    dataset_name: str = 'your_data2'         # 数据集名称
    
    # 模型参数：写自己模型需要的参数
    in_features: int = 1024 
    out_features: int = 1 
    hid_features: int = 128

    # 数据集参数: 写自己数据集需要的参数
    class_num: int = 2
    input_len: int = 1024
    data_amount: int = 100

    # 训练参数
    log_dir: str = './logs'                             # 日志目录路径
    loss: str = 'bce'                                   # 损失函数
    pretrain_checkpoint_path: Optional[str] = None      # 预训练模型路径
    
    # 监控器参数
    monitor_metric: str = 'val_loss'                    # 监控指标
    monitor_mode: Literal['max', 'min'] = 'min'         # 监控指标的模式, 也是越大越好，还是越小越好
    use_early_stopping: bool = False                    # 是否使用早停
    early_stopping_patience: int = 10                   # 早停的忍耐次数
    early_stopping_min_delta: float = 0.001             # 早停的阈值
    # 如果在 patience 轮次内，监控指标的改进小于 min_delta，则会触发早停。
    use_swa: bool = True                                # 是否使用SWA
    swa_lrs: float = 0.0005                             # swa的学习率
    swa_epoch_start: float = 0.9                        # swa的开始轮次 epochs * swa_epoch_start
    
    
    # 优化器 & 调度器
    optimizer: Literal['adam', 'sgd', 'rmsprop', 'adamw']  = 'sgd'
    weight_decay: float = 1e-5
    lr_scheduler: Literal['step', 'cosine', 'None'] = 'cosine'
    lr_decay_steps: int = 20
    lr_decay_rate: float = 0.5
    lr_decay_min_lr: float = 1e-5

    
    class Config:
        # 设置保护命名空间，避免与 Pydantic 内部的字段冲突
        protected_namespaces = ('settings_',)
    @classmethod
    def from_cli(cls):
        # Initialize parser
        parser = argparse.ArgumentParser(description="Training Configuration")
        
        # Dynamically add fields from Pydantic to argparse
        for field_name, field_type in cls.__annotations__.items():
            default_value = getattr(cls, field_name, None)
            if isinstance(field_type, type) and field_type in {str, int, float}:
                parser.add_argument(f'--{field_name}', type=field_type, default=default_value)
            elif field_type is bool:
                # Use strtobool for boolean conversion
                parser.add_argument(f'--{field_name}', type=lambda x: bool(strtobool(x)), default=default_value)
            elif hasattr(field_type, '__origin__') and field_type.__origin__ is Literal:
                # Extract literal options and create choices for argparse
                choices = [value for value in field_type.__args__]
                parser.add_argument(f'--{field_name}', choices=choices, default=default_value)
        
        # Parse command-line args and override settings
        args = parser.parse_args()
        # Convert parsed arguments to a dictionary, filtering out None values
        cli_args = {k: v for k, v in vars(args).items() if v is not None}
        
        # Use Pydantic's `.construct()` to update settings with CLI args
        return cls(**cli_args)

# Initialize settings from command-line args
train_args = TrainerSettings.from_cli()
