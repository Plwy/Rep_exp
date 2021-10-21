import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 将多个卷积核权重在输出通道维度上进行拼接, 等价于输出后拼接
def conv_cat(x):
    conv1 = nn.Conv2d(3,2,3)  # 
    conv2 = nn.Conv2d(3,4,3)  # 

    sep_output = torch.cat((conv1(x),conv2(x)),axis=1)

    weight_1 = conv1.weight.data
    weight_2 = conv2.weight.data
    conv_merge = nn.Conv2d(3,2+4,3)
    conv_merge.weight.data = torch.cat([weight_1, weight_2], axis=0)
    conv_merge.bias.data = torch.cat((conv1.bias.data, conv2.bias.data))

    merge_output = conv_merge(x)

    return sep_output, merge_output


x = np.ones(shape=(1, 3, 4, 4)).astype(np.float32)
x = torch.from_numpy(x)

original_conv, merge_conv = conv_cat(x)
original_conv, merge_conv = original_conv.detach().numpy(), merge_conv.detach().numpy()

print("cat(conv1(x),conv2(x)) output is: ", original_conv)
print("Merge output is: ", merge_conv)
print("Is Match: ", np.allclose(original_conv, merge_conv, atol=1e-5))