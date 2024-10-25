import inspect
import torch
import importlib
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl


# 基本上这个文件不需要变动
class MInterface(pl.LightningModule):
    def __init__(self, model_name, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()

    def forward(self, img):
        return self.model(img)

    def training_step(self, batch, batch_idx):
        return self.common_step(self.model, batch, self.log, self.hparams, mode='train')
    def validation_step(self, batch, batch_idx):
        return self.common_step(self.model, batch, self.log, self.hparams, mode='val')

    def test_step(self, batch, batch_idx):
        return self.common_step(self.model, batch, self.log, self.hparams, mode='test')

    def configure_optimizers(self):

        # 根据选择的优化器类型初始化优化器
        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(
                self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
            )
        else:
            raise ValueError('Invalid optimizer type!')

        if self.hparams.lr_scheduler is None or self.hparams.lr_scheduler == 'None':
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.epochs,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def load_model(self):
        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        # 读取模型模块
        try:
            Model = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!')
        
        # 读取通用推理步骤
        try:
            common_step = getattr(importlib.import_module(
                '.'+name, package=__package__), 'common_step') 
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.common_step!')
        
        self.model = self.instancialize(Model)
        self.common_step = common_step

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = list(inspect.signature(Model.__init__).parameters.keys())[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)
