#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/9 9:51
# @Author  : zqp
# @Email   : zhangqipeng@buaa.edu.cn
# @File    : model.py

# import torch
import torch.nn as nn
from model.GCN_others import MuiltKernelGTCN_
import torch.nn.functional as F
from processor.base_method import import_class


class Model(nn.Module):
    def __init__(self, n_class, graph, kernel_num, in_channels, edge_weight, lamda, graph_arg={}):
        super().__init__()
        self.n_class = n_class
        Graph = import_class(graph)
        self.graph = Graph(**graph_arg)
        self.A = self.graph.A

        kernel_size = self.A.shape[0]
        t_kernel = 9

        self.bn0 = nn.BatchNorm1d(in_channels*self.A.shape[1])  # for NTU, 3*25

        # self.backBones = nn.ModuleList((
        self.backBone1 = MuiltKernelGTCN_(in_channels, 64, self.A, False, kernel_num, (t_kernel, kernel_size), edge_weight, lamda)
        self.backBone2 = MuiltKernelGTCN_(64, 64, self.A, True, kernel_num, (t_kernel, kernel_size), edge_weight, lamda)
        self.backBone3 = MuiltKernelGTCN_(64, 64, self.A, True, kernel_num, (t_kernel, kernel_size), edge_weight, lamda)
        self.backBone4 = MuiltKernelGTCN_(64, 64, self.A, True, kernel_num, (t_kernel, kernel_size), edge_weight, lamda)
        self.backBone5 = MuiltKernelGTCN_(64, 128, self.A, True, kernel_num, (t_kernel, kernel_size), edge_weight, lamda, stride=2)
        self.backBone6 = MuiltKernelGTCN_(128, 128, self.A, True, kernel_num, (t_kernel, kernel_size), edge_weight, lamda)
        self.backBone7 = MuiltKernelGTCN_(128, 128, self.A, True, kernel_num, (t_kernel, kernel_size), edge_weight, lamda)
        self.backBone8 = MuiltKernelGTCN_(128, 256, self.A, True, kernel_num, (t_kernel, kernel_size), edge_weight, lamda, stride=2)
        self.backBone9 = MuiltKernelGTCN_(256, 256, self.A, True, kernel_num, (t_kernel, kernel_size), edge_weight, lamda)
        self.backBone10 = MuiltKernelGTCN_(256, 256, self.A, True, kernel_num, (t_kernel, kernel_size), edge_weight, lamda)
        # ))

        self.fcn = nn.Conv2d(256, n_class, kernel_size=1)  # 

    def forward(self, x):
        N, C, T, V, M = x.size()  # for NTU, (N, 3, T, 25, M)
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # for NTU, (N, M, 25, 3, T)
        x = x.view(N * M, V * C, T)  # for NTU, (N*M, 75, T)
        x = self.bn0(x)
        x = x.view(N*M, V, C, T)  #for NTU,  (N*M,25,3,T)
        x = x.permute(0, 2, 3, 1).contiguous()  # for NTU, (NM, 3, T, 25)

        
        # for net in self.backBones:
            # print(x.size())
            #　x = net(x)
        x1 = self.backBone1(x)
        x2 = self.backBone2(x1)
        x3 = self.backBone3(x2)
        x4 = self.backBone4(x3)
        x5 = self.backBone5(x4)
        x6 = self.backBone6(x5)
        x7 = self.backBone7(x6)
        x8 = self.backBone8(x7)
        x9 = self.backBone9(x8)
        x10 = self.backBone10(x9)
        

        x = F.avg_pool2d(x10, x10.size()[2:])  # pool层的大小是(T,25)
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        # （64,400）
        x = x.view(x.size(0), -1)

        return x

