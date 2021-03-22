import torch
import torch.nn as nn
import numpy as np
import math

def conv_branch_init(conv):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class MuiltKernelGCN(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 A, residual,
                 Kernel_num,
                 edge_weight,
                 Kernel_size = 3,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        """
        args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            A: 邻接矩阵
            residual: 是否使用残差
            Kernerl_num: 使用的卷积核的数量
            Kernel_siza: 数量上应等于邻接矩阵的第一维

        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.A = A
        self.Kernel_num = Kernel_num
        self.Kernel_size = Kernel_size

        assert self.A.shape[0] == self.Kernel_size # 检查Kernel_size的大小

        k, v, w = A.shape

        fixed_A = torch.tensor(A.astype(np.float32), dtype=torch.float32, requires_grad=False)
        self.register_buffer('fixed_A', fixed_A) # 注册一个buffer， 训练时不进行更新

        self.MuiltA = nn.Parameter(torch.tensor(np.reshape(A.astype(np.float32), [k, 1, v, w]),
                                                dtype=torch.float32, requires_grad=True).repeat(1, Kernel_num, 1, 1), requires_grad=True)
                                                # for NTU skeleton data, (3, 1, 25, 25)
                                                # 可训练核， 且对帧进行分组提取连接信息

        if edge_weight:
            self.edge_weight =nn.Parameter(torch.ones(self.A.shape, dtype=torch.float32), requires_grad=True)
        else:
            self.edge_weight = 1


        if not residual:
            self.residual = lambda x: 0
        elif (self.in_channels == self.out_channels):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
                nn.BatchNorm2d(self.out_channels)
            )

        self.conved = nn.Conv2d(
            self.in_channels,
            self.out_channels*self.Kernel_size,
            kernel_size=t_kernel_size,
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias
        )

        eye_array = []
        for i in range(out_channels):
            eye_array.append(torch.eye(25))
        self.eyes = nn.Parameter(torch.tensor(torch.stack(eye_array),requires_grad=False,device='cuda'),requires_grad=False) # [c,25,25]

        self.bn0 = nn.BatchNorm2d(self.out_channels*self.Kernel_size)
        self.bn = nn.BatchNorm2d(self.out_channels)
        bn_init(self.bn, 1e-6)
        self.relu = nn.ReLU()

    def norm(self, A): 
        b, c, h, w = A.size()
        A = A.view(c,25,25)
        D_list = torch.sum(A,1).view(c,1,25)
        D_list_12 = (D_list + 0.001)**(-1)
        D_12 = self.eyes * D_list_12 
        A = torch.bmm(A,D_12).view(b,c,h,w)
        return A


    def forward(self, X, lamda):
        x_residual = self.residual(X)
        device_ID = X.get_device()
        # print(device_ID)
        Kernel_A = self.MuiltA.repeat(1, self.out_channels//self.Kernel_num, 1, 1).cuda(device_ID)
        norm_kernel_A = torch.cat([self.norm(Kernel_A[0:1,...]),self.norm(Kernel_A[1:2,...]),self.norm(Kernel_A[2:3,...])],0).cuda(device_ID)
        # for NTU, (3, out_channels, 25, 25)

        x0 = self.conved(X)  # for NTU, X:(N, 3, T, 25)
        x0 = self.bn0(x0)
        n, kc, t, w = x0.size()
        x = x0.view(n, self.Kernel_size, kc//self.Kernel_size, t, w)    # for NTU, (N, k_size, out_cha, T, 25)
        
        # x1 = torch.einsum('nkctw, kwv->nctv', (x, self.fixed_A*self.edge_weight))  # 不可训练核
        # print(x.size(), Kernel_A.size())
        x2 = torch.einsum('nkctw, kcwv->nctv', (x, Kernel_A))   # 可训练卷积核得到的特征
        # x = x1 + lamda*x2
        x = self.bn(x2)
        x += x_residual
        x = self.relu(x)

        return x


class MuiltKernelGTCN(nn.Module):
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

        self.gcn = MuiltKernelGCN(self.in_channels, self.out_channels, 
                                  self.A, residual, kernel_num, 
                                  edge_weight, kernel_size[1])
        self.tcn = nn.Sequential(
            # nn.BatchNorm2d(self.out_channels),
            # nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels,
                      self.out_channels,
                      kernel_size=(kernel_size[0], 1),
                      stride=(stride, 1),
                      padding=padding),
            nn.BatchNorm2d(self.out_channels)# ,
            # nn.Dropout(dropout, inplace=True)
        )
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (self.in_channels == self.out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(kernel_size[0], 1),
                          padding=padding, stride=(stride, 1)),
                nn.BatchNorm2d(self.out_channels)
            )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        

    def forward(self, x):
        residual = self.residual(x)
        x = self.gcn(x, self.lamda)
        
        x = self.tcn(x)
        x += residual
        #　print(x.size())
        # print(residual.size())
        x = self.relu(x)
        return x




