

import torch
import torch.utils.data as data


# 方式一：继承 torch.utils.data.Dataset
# 名称要与数据集名称一致
class YourData1(data.Dataset):
    def __init__(self, 
            class_num,
            input_len,
            data_amount=100,
            train=False,
            
        ):
        self.train = train
        self.class_num = class_num
        self.input_len = input_len
        self.data_amount = data_amount

    def __len__(self):
        return self.data_amount

    def __getitem__(self, idx):
        
        y = torch.randint(0, self.class_num, (1,)).type(torch.float)
        x = 2 * y * torch.ones(self.input_len) - 1
        if self.train:
            x += torch.randn(self.input_len)
        
        return x, y