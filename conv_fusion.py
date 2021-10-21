import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# conv2_3x3(conv1_1x1(x)) = conv_merge(x)
def conv_fusion(x):
    conv1_1x1 = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=(1,1))  # 
    conv2_3x3 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(3,3))  # 
    sep_output = conv2_3x3(conv1_1x1(x))

    weight_1x1 = conv1_1x1.weight.data
    weight_3x3 = conv2_3x3.weight.data
    bias_1x1 = conv1_1x1.bias.data
    bias_3x3 = conv2_3x3.bias.data

    transpose_weight_1x1 = weight_1x1.permute(1,0,2,3)
    conv3_1x1 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=(1,1), bias=False)
    conv3_1x1.weight.data = transpose_weight_1x1    
    weight_merge = conv3_1x1(weight_3x3)
    # weight_merge = F.conv2d(weight_3x3, transpose_weight_1x1)  
    bias_merge = (weight_3x3 * bias_1x1.reshape(1,-1,1,1)).sum((1,2,3)) + bias_3x3

    conv_merge = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(3,3))
    # conv_merge.weight.data = weight_merge
    # conv_merge.bias.data = bias_merge
    conv_merge.weight.data = weight_merge
    conv_merge.bias.data = bias_merge
    merge_output = conv_merge(x)

    return sep_output, merge_output


x = np.ones(shape=(1, 3, 4, 4)).astype(np.float32)
x = torch.from_numpy(x)

original_conv, merge_conv = conv_fusion(x)
original_conv, merge_conv = original_conv.detach().numpy(), merge_conv.detach().numpy()

print("conv2_3x3(conv1_1x1(x)) output is: ", original_conv)
print("Merge output is: ", merge_conv)
print("Is Match: ", np.allclose(original_conv, merge_conv, atol=1e-5))