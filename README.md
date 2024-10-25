# PyTorch Lightning 训练模板

- 此仓库提供了一个基于 PyTorch Lightning 的机器学习模型训练的结构化模板。旨在简化设置训练工作流、组织代码和管理配置的过程。

# 目录结构

```
root/
	├── data/
	│   ├── __init__.py
	│   ├── aug.py                  # 数据增强相关方法
	│   ├── data_interface.py       # 数据集接口模板（一般不需要修改）
	│   ├── your_data1.py           # 自定义数据集 1
	│   ├── your_data2.py           # 自定义数据集 2
	│   └── ...
	├── model/
	│   ├── __init__.py
	│   ├── common.py               # 基础网络组件
	│   ├── model_interface.py       # 模型接口模板（一般不需要修改）
	│   ├── your_net1.py            # 自定义模型 1（必须包含模型定义和 common_step 方法）
	│   ├── your_net2.py            # 自定义模型 2
	│   └── ...
	├── config.py                   # 配置文件
	├── run_train.sh                # 训练脚本
	├── train.py                    # 主训练文件
	└── utils.py                    # 工具函数
```

## 快速开始

### 1. 创建数据集
- 在 `data` 文件夹中创建一个新的数据集文件，例如 `your_data1.py`。
- 您可以选择继承 `torch.utils.data.Dataset`（例如在 `your_data1.py` 中），或创建 `torch.utils.data.TensorDataset`（例如在 `your_data2.py` 中）。

### 2. 定义模型
- 在 `model` 文件夹中创建一个新的模型文件，例如 `your_net1.py`。
- 在此文件中实现模型架构。确保包含 `common_step` 方法，以计算损失并记录指标，以便进行反向传播。

### 3. 配置参数
- 编辑 `config.py` 文件以设置模型参数和数据集配置。通过这种方式可以更容易地管理和调整训练过程中的参数。

### 4. 运行训练
- 使用提供的 `run_train.sh` 脚本来启动训练过程。此脚本将设置环境并执行训练文件。


### 数据集示例（`your_data1.py`）

```python
import torch
from torch.utils.data import Dataset

class YourDataset(Dataset):
    def __init__(self, data_path):
        # 加载您的数据
        self.data = ...
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 返回单个数据点
        return self.data[idx]
```

### 模型示例（`your_net1.py`）

```python
from torch import nn
from torch.nn import functional as F
from config import TrainerSettings

# 你的模型
class YourNet(nn.Module):
    def __init__(self, in_features=1024, out_features=1, hid_features=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, hid_features),
            nn.ReLU(inplace=True),
            nn.Linear(hid_features, out_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        return x


def common_step(model, batch, log, hparams: TrainerSettings, mode: str='train'):
    """
    这里写推理代码
    """
    # 提取数据
    img, labels = batch
    out = model(img)

    # 计算loss
    loss = F.binary_cross_entropy_with_logits(out, labels)    
    # 计算acc
    acc = (out > 0.5).eq(labels).float().mean()

    # 记录到tensorboard
    log(f'{mode}_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    # 记录到tensorboard
    log(f'{mode}_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

    # 返回loss
    return loss
```
