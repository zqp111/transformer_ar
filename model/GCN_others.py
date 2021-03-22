import torch
import numpy as np
import torch.nn as nn
import math

def conv_branch_init(conv):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal(weight, 0, math.sqrt(2. / (n * k1 * k2)))
    nn.init.constant(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal(conv.weight, mode='fan_out')
    nn.init.constant(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant(bn.weight, scale)
    nn.init.constant(bn.bias, 0)




class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, groups, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.A = A
        self.groups = groups
        self.num_subset = num_subset

        fixed_A = torch.tensor(A.astype(np.float32), dtype=torch.float32, requires_grad=False)
        self.register_buffer('fixed_A', fixed_A) 
        
        self.DecoupleA = nn.Parameter(torch.tensor(np.reshape(A.astype(np.float32),[3,1,25,25]), dtype=torch.float32, requires_grad=True).repeat(1,groups,1,1), requires_grad=True)
        

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn0 = nn.BatchNorm2d(out_channels * num_subset)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

        self.Linear_weight = nn.Parameter(torch.zeros(in_channels,out_channels * num_subset,requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0,math.sqrt(0.5/ (out_channels * num_subset)))

        self.Linear_bias = nn.Parameter(torch.zeros(1,out_channels * num_subset,1,1,requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant(self.Linear_bias, 1e-6)

        eye_array = []
        for i in range(out_channels):
            eye_array.append(torch.eye(25))
        self.eyes = nn.Parameter(torch.tensor(torch.stack(eye_array),requires_grad=False,device='cuda'),requires_grad=False) # [c,25,25]
        self.edge_weight =nn.Parameter(torch.ones(self.A.shape, dtype=torch.float32), requires_grad=True)


    def norm(self, A): 
        b, c, h, w = A.size()
        A = A.view(c,25,25)
        D_list = torch.sum(A,1).view(c,1,25)
        D_list_12 = (D_list + 0.001)**(-1)
        D_12 = self.eyes * D_list_12 
        A = torch.bmm(A,D_12).view(b,c,h,w)
        return A

    def forward(self, x0):
        learn_A = self.DecoupleA.repeat(1,self.out_channels//self.groups,1,1)
        norm_learn_A = torch.cat([self.norm(learn_A[0:1,...]),self.norm(learn_A[1:2,...]),self.norm(learn_A[2:3,...])],0)

        x = torch.einsum('nctw,cd->ndtw', (x0, self.Linear_weight)).contiguous()
        x = x + self.Linear_bias
        x = self.bn0(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.num_subset, kc//self.num_subset, t, v)
        x1 = torch.einsum('nkctw, kwv->nctv', (x, self.fixed_A*self.edge_weight))  # 不可训练核
        x2 = torch.einsum('nkctv,kcvw->nctw', (x, norm_learn_A))
        x = x1 + 0.5*x2

        x = self.bn(x)
        x += self.down(x0)
        x = self.relu(x)
        return x


class MuiltKernelGTCN_(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 A, residual,
                 kernel_num,
                 kernel_size,
                 edge_weight,
                 lamda,
                 stride=1,
                 dropout=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.A = A
        self.lamda = lamda
        assert len(kernel_size) == 2
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = unit_gcn(self.in_channels, self.out_channels, self.A, kernel_num)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels,
                      self.out_channels,
                      kernel_size=(kernel_size[0], 1),
                      stride=(stride, 1),
                      padding=padding),
            nn.BatchNorm2d(self.out_channels),
            nn.Dropout(dropout, inplace=True)
        )
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif self.in_channels == self.out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.out_channels)
            )

    def forward(self, x):
        residual = self.residual(x)
        x = self.gcn(x)
        x += residual
        x = self.tcn(x)
        x = self.relu(x)
        return x