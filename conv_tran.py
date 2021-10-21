import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# tran conv_1x1 to conv_3x3
def conv_tran_1x1(x):
    kernel_size = (1,1)
    conv_1x1 = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=kernel_size, bias=False)  # 
    ori_output = conv_1x1(x)
    target_kernel_size = (3,3)
   
    H_pixels_to_pad = (target_kernel_size[0] - kernel_size[0]) // 2
    W_pixels_to_pad = (target_kernel_size[1] - kernel_size[0]) // 2
    weight_1x1 = conv_1x1.weight        # (2,3,1,1)
    weight_1x1_tran = F.pad(weight_1x1, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])   # (2,3,3,3)
    conv_1x1_tran = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=target_kernel_size, bias=False)
    conv_1x1_tran.weight.data = weight_1x1_tran

    x = F.pad(x, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad]) # input pad too!
    tran_output = conv_1x1_tran(x)
    print(tran_output.shape)
    return ori_output, tran_output

x = np.ones(shape=(1, 3, 4, 4)).astype(np.float32)
x = torch.from_numpy(x)

ori_output, tran_output = conv_tran_1x1(x)
ori_output, tran_output = ori_output.detach().numpy(), tran_output.detach().numpy()
print("======1x1 tran 3x3=====")
print("conv_1x1(x) output is: ", ori_output)
print("tran conv_3x3 output is: ", tran_output)
print("Is Match: ", np.allclose(ori_output, tran_output, atol=1e-5))

