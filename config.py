from pydantic_settings import BaseSettings
from typing import Literal, Optional
import argparse

class TrainerSettings(BaseSettings):
    # 基础参数
    seed: int = 159456                      # 随机种子
    epochs: int = 100                       # 训练轮数
    batch_size: int = 32                    # 训练批次数
    num_workers: int = 4                    # 训练数据加载线程数
    lr: float = 1e-3                        # 学习率
    model_name: str = 'your_net'            # 模型名称
    dataset_name: str = 'your_data'         # 数据集名称
    

    # 模型参数
    hidden_units: int = 64  # Changed `hid` to `hidden_units`
    input_channels: int = 1024  # Changed `in_channel` to `input_channels`
    layer_count: int = 5  # Changed `layer_num` to `layer_count`

    # 数据集参数
    class_num: int = 9
    input_size: tuple[int, int] = (224, 224)

    # 训练参数
    log_dir: str = './logs'                             # 日志目录路径
    loss: str = 'bce'                                   # 损失函数
    pretrain_checkpoint_path: Optional[str] = None      # 预训练模型路径
    
    # 监控器参数
    monitor_metric: str = 'val_loss'                    # 监控指标
    monitor_mode: Literal['max', 'min'] = 'min'         # 监控指标的模式, 也是越大越好，还是越小越好
    use_early_stopping: bool = False                    # 是否使用早停
    early_stopping_patience: int = 10                   # 早停的忍耐次数
    early_stopping_min_delta: float = 0.0001            # 早停的阈值
    # 如果在 patience 轮次内，监控指标的改进小于 min_delta，则会触发早停。
    
    # 优化器 & 调度器
    optimizer: Literal['adam', 'sgd', 'rmsprop', 'adamw']  = 'sgd'
    weight_decay: float = 1e-5
    lr_scheduler: Literal['step', 'cosine', None] = 'cosine'
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
            if isinstance(default_value, bool):
                # Special handling for boolean flags
                parser.add_argument(f'--{field_name}', action='store_true' if not default_value else 'store_false')
            else:
                parser.add_argument(f'--{field_name}', type=field_type, default=default_value)
        
        # Parse command-line args and override settings
        args = parser.parse_args()
        # Convert parsed arguments to a dictionary, filtering out None values
        cli_args = {k: v for k, v in vars(args).items() if v is not None}
        
        # Use Pydantic's `.construct()` to update settings with CLI args
        return cls(**cli_args)

# Initialize settings from command-line args
train_args = TrainerSettings.from_cli()
