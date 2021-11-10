import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

def _fuse_bn_tensor(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


if __name__ == '__main__':
    # contruct input
    # x = np.ones(shape=(3, 3, 4, 4)).astype(np.float32)
    x = np.random.randint(0,100,(1, 3, 4, 4)).astype(np.float32)
    x = torch.from_numpy(x)

    in_planes = 3
    out_planes = 8
    stride = 1
    conv0 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    bn0 = nn.BatchNorm2d(out_planes)

    print(conv0(x).mean())


    print(bn0.running_mean, bn0.running_var,bn0.weight, bn0.bias, bn0.eps)

    ori_output1 = bn0(conv0(x))
    ori_output1 = ori_output1.detach().numpy()

    
    """eval() very important"""
    conv0.eval()
    bn0.eval()
    print(bn0.running_mean, bn0.running_var,bn0.weight, bn0.bias, bn0.eps)
    ori_output = bn0(conv0(x))

    # fusion1
    conv0_bn0_f1 = torch.nn.utils.fuse_conv_bn_eval(conv0, bn0).eval()
    f_out1 = conv0_bn0_f1(x)

    # fusion2
    k0, b0 = _fuse_bn_tensor(conv0, bn0)
    conv0_bn0_f2 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1)
    conv0_bn0_f2.weight.data = k0
    conv0_bn0_f2.bias.data = b0
    f_out2 = conv0_bn0_f2(x)

    # compare
    ori_output, f_out1, f_out2 = ori_output.detach().numpy(), f_out1.detach().numpy(), f_out2.detach().numpy()
    print(" f_out1, f_out2 Is Match: ", np.allclose(f_out1, f_out2, atol=1e-5))
    print("tran diff: ", ((f_out1 - f_out2)**2).sum())


    print(" ori_output, f_out1 Is Match: ", np.allclose(ori_output, f_out1, atol=1e-5))
    print("tran diff: ", ((ori_output - f_out1)**2).sum())


    print(" ori_output, f_out2 Is Match: ", np.allclose(ori_output, f_out2, atol=1e-5))
    print("tran diff: ", ((ori_output - f_out2)**2).sum())

    print(" ori_output, ori_output1 Is Match: ", np.allclose(ori_output, ori_output1, atol=1e-5))
    print(((ori_output - ori_output1)**2).sum())