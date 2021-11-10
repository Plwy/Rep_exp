import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class RepBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):

        super(RepBlock, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride

        self.conv33 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn33 = nn.BatchNorm2d(out_planes)
        self.conv11 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn11 = nn.BatchNorm2d(out_planes)
        if self.in_planes == self.out_planes and self.stride == 1:
            self.bn00 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = self.bn33(self.conv33(x))
        out += self.bn11(self.conv11(x))
        if self.in_planes == self.out_planes and self.stride == 1:
            out += self.bn00(x)

        # return out
        return F.relu(out)

    # ori
    def deploy(self):
        self.eval()
        conv33_bn33 = torch.nn.utils.fuse_conv_bn_eval(self.conv33, self.bn33).eval()
        conv11_bn11 = torch.nn.utils.fuse_conv_bn_eval(self.conv11, self.bn11).eval()
        conv33_bn33.weight.data += F.pad(conv11_bn11.weight.data, [1, 1, 1, 1])
        conv33_bn33.bias.data += conv11_bn11.bias.data
        if self.in_planes == self.out_planes and self.stride == 1:
            conv00 = nn.Conv2d(self.in_planes, self.out_planes, kernel_size=3, padding=1, bias=False).eval()
            nn.init.dirac_(conv00.weight.data)
            conv00_bn00 = torch.nn.utils.fuse_conv_bn_eval(conv00, self.bn00)
            conv33_bn33.weight.data += conv00_bn00.weight.data
            conv33_bn33.bias.data += conv00_bn00.bias.data
        return [conv33_bn33,nn.ReLU(True)]

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


    # def deploy(self, x):
    #     self.eval()
    #     conv33_bn33 = torch.nn.utils.fuse_conv_bn_eval(self.conv33, self.bn33).eval()
    #     conv11_bn11 = torch.nn.utils.fuse_conv_bn_eval(self.conv11, self.bn11).eval()
    #     conv33_bn33.weight.data += F.pad(conv11_bn11.weight.data, [1, 1, 1, 1])
    #     conv33_bn33.bias.data += conv11_bn11.bias.data

    #     ## 1. rep 中的转换是否等于 dirac 表示的转换 ?   是
    #     ## 2.判断 bn(x) 是否等于 conv00_bn00(x)
    #     if self.in_planes == self.out_planes and self.stride == 1:

    #         """dirac"""
    #         conv00_1 = nn.Conv2d(self.in_planes, self.out_planes, kernel_size=3, padding=1, bias=False).eval()
    #         nn.init.dirac_(conv00_1.weight.data)
    #         conv00_bn00_1 = torch.nn.utils.fuse_conv_bn_eval(conv00_1, self.bn00)
    #         # conv33_bn33.weight.data += conv00_bn00.weight.data
    #         # conv33_bn33.bias.data += conv00_bn00.bias.data

    #         """rep"""
    #         if not hasattr(self, 'id_tensor'):
    #             self.group = 1
    #             input_dim = self.in_planes // self.group
    #             kernel_value = np.zeros((self.in_planes, input_dim, 3, 3), dtype=np.float32)
    #             for i in range(self.in_planes):  
    #                 kernel_value[i, i % input_dim, 1, 1] = 1

    #             # self.id_tensor = torch.from_numpy(kernel_value)
    #         conv00_2 = nn.Conv2d(self.in_planes, self.out_planes, kernel_size=3, padding=1, bias=False).eval()
    #         conv00_2.weight.data = torch.from_numpy(kernel_value)

    #         conv00_bn00_2 = torch.nn.utils.fuse_conv_bn_eval(conv00_2, self.bn00)

    #         if conv00_bn00_2.weight.data.equal(conv00_bn00_1.weight.data):
    #             print("=======equal !!==========")  

    #         self.bn00.eval()
    #         test1 = self.bn00(x) 
    #         test2 = conv00_1(x)
    #         test3 = conv00_bn00_1(x)
            
    #         if test1.equal(test2):
    #             print("test1.equal(test2)")
    #         else:
    #             print((test1-test2).sum())
    #         if test2.equal(test3):
    #             print("test2.equal(test3)")

    #         if test1.equal(test3):
    #             print("test1.equal(test3)")
    #         else:
    #             print((test1-test3).sum())

    #     out = conv33_bn33(x) + conv00_bn00_1(x)

    #     # return [conv33_bn33,nn.ReLU(True)]
    #     return out
    #     # return F.relu(out)


    # def deploy(self, x):
    #     # self.eval()
    #     with torch.no_grad():
    #         conv33_bn33 = torch.nn.utils.fuse_conv_bn_eval(self.conv33, self.bn33).eval()
    #         conv11_bn11 = torch.nn.utils.fuse_conv_bn_eval(self.conv11, self.bn11).eval()
    #         # conv33_bn33.weight.data += F.pad(conv11_bn11.weight.data, [1, 1, 1, 1])

    #         k33, b33 = self._fuse_bn_tensor(self.conv33, self.bn33)
    #         k11, b11 = self._fuse_bn_tensor(self.conv11, self.bn11)
    #         print("对比torch 方法的实现33")
    #         if conv33_bn33.weight.data.equal(k33):
    #             print('k weight equal!')
    #         else:
    #             print('k weight not equal!')

    #         if conv33_bn33.bias.data.equal(b33):
    #             print('bias equal!')
    #         else:
    #             print('bias not equal!')
    #             print(b33)
    #             print(conv33_bn33.bias.data)

    #         print("对比torch 方法的实现11")
    #         if conv11_bn11.weight.data.equal(k11):
    #             print('k weight equal!')
    #         else:
    #             print('k weight not equal!')

    #         if conv11_bn11.bias.data.equal(b11):
    #             print('bias equal!')
    #         else:
    #             print('bias not equal!')
    #             print(b11)
    #             print(conv11_bn11.bias.data)

    #     conv33_bn33.weight.data += torch.nn.functional.pad(conv11_bn11.weight.data, [1,1,1,1])
    #     conv33_bn33.bias.data += conv11_bn11.bias.data

    #     """对identity bn做处理"""
    #     if self.in_planes == self.out_planes and self.stride == 1:
    #         #"""ori repVGG实现"""
    #         # if not hasattr(self, 'id_tensor'):
    #         #     self.group = 1
    #         #     input_dim = self.in_planes // self.group
    #         #     kernel_value = np.zeros((self.in_planes, input_dim, 3, 3), dtype=np.float32)
    #         #     for i in range(self.in_planes):  
    #         #         kernel_value[i, i % input_dim, 1, 1] = 1

    #         #     # self.id_tensor = torch.from_numpy(kernel_value)
    #         # conv00 = nn.Conv2d(self.in_planes, self.out_planes, kernel_size=3, padding=1, bias=False).eval()
    #         # conv00.weight.data = torch.from_numpy(kernel_value)

    #         # conv00_bn00 = torch.nn.utils.fuse_conv_bn_eval(conv00, self.bn00)
    #         # conv33_bn33.weight.data += conv00_bn00.weight.data
    #         # conv33_bn33.bias.data += conv00_bn00.bias.data

    #         """REMNet dirac init方法"""
    #         conv00 = nn.Conv2d(self.in_planes, self.out_planes, kernel_size=3, padding=1, bias=False).eval()
    #         nn.init.dirac_(conv00.weight.data)
    #         conv00_bn00 = torch.nn.utils.fuse_conv_bn_eval(conv00, self.bn00)
    #         conv33_bn33.weight.data += conv00_bn00.weight.data
    #         conv33_bn33.bias.data += conv00_bn00.bias.data
        
    #     out = F.relu(conv33_bn33(x))

    #     return out


class RMBlock(nn.Module):
    def __init__(self, planes, ratio=0.5):

        super(RMBlock, self).__init__()
        self.planes = planes
        self.rmplanes=int(planes*ratio)
        
        self.conv33 = nn.Conv2d(planes, self.planes-self.rmplanes, kernel_size=3, padding=1, bias=False)
        self.bn33 = nn.BatchNorm2d(self.planes-self.rmplanes)
        self.conv11 = nn.Conv2d(planes, self.planes-self.rmplanes, kernel_size=1, padding=0, bias=False)
        self.bn11 = nn.BatchNorm2d(self.planes-self.rmplanes)
        self.bn00 = nn.BatchNorm2d(self.planes-self.rmplanes)

    def forward(self, x):
        out = self.bn33(self.conv33(x))
        out += self.bn11(self.conv11(x))
        out += self.bn00(x[:,self.rmplanes:])
        return F.relu(torch.cat([x[:,:self.rmplanes],out],dim=1))
    
    def deploy(self):
        self.eval()
        conv33=nn.utils.fuse_conv_bn_eval(self.conv33,self.bn33)
        conv11=nn.utils.fuse_conv_bn_eval(self.conv11,self.bn11)
        conv00=nn.Conv2d(self.planes,self.planes-self.rmplanes,kernel_size=3,padding=1,bias=False).eval()
        nn.init.zeros_(conv00.weight.data[:,:self.rmplanes])
        nn.init.dirac_(conv00.weight.data[:,self.rmplanes:])
        conv00=nn.utils.fuse_conv_bn_eval(conv00,self.bn00)
        conv3=nn.Conv2d(self.planes,self.planes,kernel_size=3,padding=1)
        conv1=nn.Conv2d(self.planes,self.planes,kernel_size=1)
        conv0=nn.Conv2d(self.planes,self.planes,kernel_size=3,padding=1)
        nn.init.zeros_(conv3.weight.data[:self.rmplanes])
        nn.init.zeros_(conv1.weight.data[:self.rmplanes])
        nn.init.dirac_(conv0.weight.data[:self.rmplanes])
        nn.init.zeros_(conv3.bias.data[:self.rmplanes])
        nn.init.zeros_(conv1.bias.data[:self.rmplanes])
        nn.init.zeros_(conv0.bias.data[:self.rmplanes])
        conv3.weight.data[self.rmplanes:]=conv33.weight.data
        conv1.weight.data[self.rmplanes:]=conv11.weight.data
        conv0.weight.data[self.rmplanes:]=conv00.weight.data
        conv3.bias.data[self.rmplanes:]=conv33.bias.data
        conv1.bias.data[self.rmplanes:]=conv11.bias.data
        conv0.bias.data[self.rmplanes:]=conv00.bias.data
        
        conv3.weight.data += F.pad(conv1.weight.data, [1, 1, 1, 1])
        conv3.bias.data += conv1.bias.data
        conv3.weight.data += conv0.weight.data
        conv3.bias.data += conv0.bias.data
        return [conv3,nn.ReLU(True)]


class RMRep(nn.Module):
    def __init__(self, num_blocks, num_classes=1000, base_wide=64,ratio=0.5):
        super(RMRep, self).__init__()
        feature=[]
        in_planes=3
        for b,t,s,n in num_blocks:
            out_planes=t*base_wide
            for i in range(n):
                if b=='rm_rep':
                    feature.append(RMBlock(out_planes,ratio))
                feature.append(RepBlock(in_planes,out_planes,s))
                in_planes=out_planes
                
        self.feature=nn.Sequential(*feature)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.flat = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(base_wide*num_blocks[-1][1], num_classes)

    def forward(self, x):
        out = self.feature(x)
        out = self.gap(out)
        out = self.flat(out)
        out = self.fc(out)
        return out
    
    def deploy(self):
        blocks=[]
        for m in self.feature:
            if isinstance(m,RepBlock) or isinstance(m,RMBlock):
                blocks+=m.deploy()
        blocks.append(self.gap)
        blocks.append(self.flat)
        blocks.append(self.fc)
        return nn.Sequential(*blocks)

def repvgg_69(num_classes=1000,depth=8):     
    return RMRep([['rep',1,2,2] if num_classes==1000 else ['rep',1,1,1],
                   ['rm_rep',1,1,depth],
                   ['rep_rep',2,2,1],
                   ['rm_rep',2,1,depth],
                   ['rep',4,2,1],
                   ['rm_rep',4,1,depth],
                   ['rep',8,2,1],
                   ['rm_rep',8,1,depth]],
                   num_classes=num_classes,ratio=0)

class Simple_rep(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1):

        super(Simple_rep, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes

        self.conv_init = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        nn.init.constant(self.conv_init.weight.data, 3)

        self.rep_vgg = RepBlock(8, 8)

    
    def forward(self, x):
        out = self.conv_init(x)
        out = self.rep_vgg(out)
        return out

    def deploy(self, x):
        out = self.conv_init(x)
        layers = self.rep_vgg.deploy()
        out = layers[0](out)
        out = layers[1](out)

        # out = self.rep_vgg.deploy(out)

        return out




class Simple_rem(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1):

        super(Simple_rem, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        # self.stride = stride

        self.conv_init = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        nn.init.constant(self.conv_init.weight.data, 3)

        self.rem_vgg = RMBlock(8, ratio=0)
    
    def forward(self, x):
        out = self.conv_init(x)
        out = self.rem_vgg(out)
        return out

    def deploy(self, x):
        out = self.conv_init(x)
        # out = self.rem_vgg.deploy(out)

        layers = self.rem_vgg.deploy()
        out = layers[0](out)
        out = layers[1](out)

        return out

if __name__ == '__main__':
    model_rep = Simple_rep(3,8)
    model_rem = Simple_rem(3,8)

    # contruct input
    x = np.ones(shape=(1, 3, 4, 4)).astype(np.float32)
    x = torch.from_numpy(x)

    """
    rep 和rem对比, reserving ratio set 0
    """
    model_rep.eval()
    model_rem.eval()
    rep_output = model_rep(x)
    rem_output = model_rem(x)
    rep_output, rem_output = rep_output.detach().numpy(), rem_output.detach().numpy()
    print("Is Match: ", np.allclose(rep_output, rem_output, atol=1e-5))
    print("tran diff: ", ((rep_output - rem_output)**2).sum())

    """
        rep 对比 
    """
    model_rep.eval()
    rep_output1 = model_rep(x)
    rep_output2 = model_rep.deploy(x)
    rep_output1, rep_output2 = rep_output1.detach().numpy(), rep_output2.detach().numpy()
    print("Is Match: ", np.allclose(rep_output1, rep_output2, atol=1e-5))
    print("tran diff: ", ((rep_output1 - rep_output2)**2).sum())


    """
        rem 对比
    """
    model_rem.eval()
    rem_output1 = model_rem(x)
    rem_output2 = model_rem.deploy(x)
    rep_output1, rem_output2 = rem_output1.detach().numpy(), rem_output2.detach().numpy()
    print("Is Match: ", np.allclose(rep_output1, rem_output2, atol=1e-5))
    print("tran diff: ", ((rep_output1 - rem_output2)**2).sum())