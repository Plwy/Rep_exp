import numpy as np
import torch
import torch.nn as nn

## conv1(x)+conv2(x) = conv3(x)
def conv_add(x):
    conv1 = nn.Conv2d(3,8,(3,3))
    conv2 = nn.Conv2d(3,8,(3,3))
    
    conv_merge = nn.Conv2d(3,8,(3,3))
    conv_merge.weight.data = conv1.weight.data + conv2.weight.data
    conv_merge.bias.data = conv1.bias.data + conv2.bias.data
    return conv1(x)+conv2(x), conv_merge(x)

# conv_1x3(x) + conv_3x3(x) = conv_merge(x)
def conv_1x3_add_3x3(x):
    # conv_1x3 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(1,3), padding=[0,1], bias=False)  # padding保证输出尺寸一致
    # conv_3x3 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(3,3), padding=[1,1], bias=False) 
    conv_1x3 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(1,3), padding=[0,1], padding_mode='zeros')  # padding保证输出尺寸一致
    conv_3x3 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(3,3), padding=[1,1], padding_mode='zeros') 
    # conv_1x3 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(1,3), padding_mode='zeros')  # padding保证输出尺寸一致
    # conv_3x3 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(3,3), padding_mode='zeros') 
    print(conv_1x3(x).shape)
    print(conv_3x3(x).shape)

    ori_output = conv_1x3(x)+conv_3x3(x)
    print('ori_output shape:',ori_output.shape )

    conv_merge = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(3,3), padding=[1,1], padding_mode='zeros') 
    # conv_merge = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(3,3)) 


    weight_1x3 = conv_1x3.weight.data # (4,3,1,3)
    weight_3x3 = conv_3x3.weight.data # (4,3,3,3)

    weight_merge = weight_3x3
    weight_merge[:,:,1:2,0:3] += weight_1x3

    bias_merge = conv_1x3.bias.data + conv_3x3.bias.data
    conv_merge.weight.data = weight_merge
    conv_merge.bias.data = bias_merge

    merge_output = conv_merge(x)
    print('merge_output shape:',merge_output.shape )

    return ori_output, merge_output

x = np.ones(shape=(1, 3, 4, 4)).astype(np.float32)
x = torch.from_numpy(x)

# original_conv_add, merge_conv_add = conv_add(x)
# original_conv_add, merge_conv_add = original_conv_add.detach().numpy(), merge_conv_add.detach().numpy()
# print("======3x3 add 3x3=====")
# print("Conv1 + Conv2 output is: ", original_conv_add)
# print("Merge Add output is: ", merge_conv_add)
# print("Is Match: ", np.allclose(original_conv_add, merge_conv_add, atol=1e-5))

ori_output, merge_output = conv_1x3_add_3x3(x)
ori_output, merge_output = ori_output.detach().numpy(), merge_output.detach().numpy()
print("======1x3 add 3x3=====")
print(" Conv1 + Conv2 output is:  ", ori_output)
print("add to output is: ", merge_output)
print("Is Match: ", np.allclose(ori_output, merge_output, atol=1e-5))