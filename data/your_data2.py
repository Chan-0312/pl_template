import torch
from torch.utils.data import TensorDataset

# 方式二：可以直接返回tensordataset类型
# 名称要与数据集名称一致
def YourData2(class_num, input_len, data_amount=100, train=False):
       
    y = torch.randint(0, class_num, (data_amount, 1)).type(torch.float)
 
    if train:
        x = 2 * y * torch.ones((data_amount, input_len)) - 1 + torch.randn((data_amount, input_len))
    else:
        x = 2 * y * torch.ones((data_amount, input_len)) - 1 
        
    
    dataset = TensorDataset(x, y)

    return dataset