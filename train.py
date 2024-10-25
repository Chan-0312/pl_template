
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger
from model import MInterface
from data import DInterface
from utils import logger
from config import TrainerSettings, train_args 

torch.set_float32_matmul_precision('high')

def main(args:TrainerSettings):
    # 设置随机种子
    logger.info(f'设置随机种子为：{args.seed}')
    logger.info(f'训练模型名称: {args.model_name}')
    logger.info(f'使用数据集：{args.dataset_name}')
    pl.seed_everything(args.seed, verbose=False)
    
    # 判断是否要加载预训练模型
    if args.pretrain_checkpoint_path is not None:
        if not os.path.exists(args.pretrain_checkpoint_path):
            args.pretrain_checkpoint_path = None
            logger.warning(f'预训练模型不存在，将使用随机初始化模型！')
            model = MInterface(**vars(args))
        else:
            model = MInterface.load_from_checkpoint(args.pretrain_checkpoint_path, **vars(args))
    else:
        model = MInterface(**vars(args))

    # 初始化数据集
    data_module = DInterface(**vars(args))

    # 监控器
    callbacks = []
    if args.use_early_stopping:
        callbacks.append(plc.EarlyStopping(
            monitor=args.monitor_metric,
            mode=args.monitor_mode,
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta
        ))
    
    filename = 'checkpoint-{epoch:02d}-{%s:.3f}'%args.monitor_metric
    callbacks.append(plc.ModelCheckpoint(
        monitor=args.monitor_metric,
        mode=args.monitor_mode,
        filename=filename,
        save_top_k=3,       # 只保存最好的3个模型  
        save_last=True      # 保留最后一个epoch的模型
    ))

    # 训练器初始化
    trainer = Trainer(
        logger = TensorBoardLogger(
            save_dir=args.log_dir, 
            name=args.model_name
        ), 
        callbacks = callbacks,
        max_epochs = args.epochs
    )

    # 训练模型
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    
    main(train_args)
