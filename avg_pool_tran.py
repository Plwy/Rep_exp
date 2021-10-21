import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def avg_pool_test(x):
    avg_pool = nn.AvgPool2d(kernel_size=3,stride=1,padding=0)
    ori_output = avg_pool(x)
    print(ori_output.shape)

    conv_avg = nn.Conv2d(3,3,kernel_size=3, stride=1)
    weight = torch.zeros((3, 3, 3, 3))

    # 对当前输入通道设置固定的权重，对其他通道权重设置为0
    weight[np.arange(3),np.arange(3),:,:] = 1 / 3 ** 2
    # 2.
    # for i in range(3):
    #     k  = i + 1 
    #     if k > 2:
    #         k = 0
    #     weight[i, k, :, :] = 1 /9
    
    conv_avg.weight.data = weight
    conv_avg.bias.data = torch.zeros(3)

    tran_output = conv_avg(x)
    print(tran_output.shape)

    return ori_output, tran_output


x = np.ones(shape=(1, 3, 4, 4)).astype(np.float32)
x = torch.from_numpy(x)

ori_output, tran_output = avg_pool_test(x)
ori_output, tran_output = ori_output.detach().numpy(), tran_output.detach().numpy()

print("avg_pool output is: ", ori_output)
print("tran conv output is: ", tran_output)
print("Is Match: ", np.allclose(ori_output, tran_output, atol=1e-5))

