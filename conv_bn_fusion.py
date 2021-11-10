import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# merge conv and bn
def conv_bn_fusion(x):
    conv_3x3 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, bias=False)  # 
    # conv_3x3 = nn.Conv2d(3,4,3)  # 
    # conv_3x3.bias.data = torch.zeros(4)

    bn = nn.BatchNorm2d(num_features=4, affine=True)
    nn.init.uniform_(bn.running_mean, 0, 0.1)
    nn.init.uniform_(bn.running_var, 0, 0.1)
    nn.init.uniform_(bn.weight, 0, 0.1)
    nn.init.uniform_(bn.bias, 0, 0.1)
    conv_3x3.eval()
    bn.eval()

    sep_output = bn(conv_3x3(x))
    print('sepout', sep_output.shape)


    weight_3x3 = conv_3x3.weight
    print('3x3:',weight_3x3.shape)

    std = (bn.running_var + bn.eps).sqrt()
    conv_merge = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3)
    t = (bn.weight / std).reshape(-1, 1, 1, 1)
    weight_merge = conv_3x3.weight * t
    conv_merge.weight.data = weight_merge
    conv_merge.bias.data = bn.bias - bn.running_mean * bn.weight / std

    merge_output = conv_merge(x)
    print('merge_output', merge_output.shape)

    return sep_output, merge_output


x = np.ones(shape=(1, 3, 4, 4)).astype(np.float32)
x = torch.from_numpy(x)

sep_output, merge_output = conv_bn_fusion(x)
sep_output, merge_output = sep_output.detach().numpy(), merge_output.detach().numpy()

print("bn(conv_3x3(x)) output is: ", sep_output)
print("Merge output is: ", merge_output)
print("conv+bn tran conv_merge diff: ", ((merge_output - sep_output)**2).sum())