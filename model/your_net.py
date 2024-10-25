from torch import nn
from torch.nn import functional as F
from config import TrainerSettings

# 你的模型
class YourNet(nn.Module):
    def __init__(self, in_channel=1024, out_channel=1, hid=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channel, hid),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(hid, out_channel),
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
    loss = F.binary_cross_entropy(out, labels)    

    # 记录到tensorboard
    log(f'{mode}_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    # 返回loss
    return loss