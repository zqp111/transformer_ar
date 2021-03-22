## -*- encoding: utf-8 -*-
'''
@File    :   TR_model.py
@Time    :   2021/03/05 15:53:55
@Author  :   zqp 
@Version :   1.0
@Contact :   zhangqipeng@buaa.edu.cn
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/home2/zqp/TR-AR')
# print(sys.path)
from modules.Encoder import Encoder
from modules.graph_attention import MultiHeadedGraphAttention
from processor.base_method import import_class
from model.GCN_others import MuiltKernelGTCN_


class Model(nn.Module):
    def __init__(self,
                graph,
                kernel_num,
                edge_weight,
                lamda,

                input_channel,
                mid_channels: list,
                layer_num,
                position_encoding, 
                heads, 
                encode_size, 
                d_ff, 
                dropout, 
                attention_dropout, 
                position_encoding_dropout,
                point_num = 25, n_classes = 60, graph_arg={}):
        super(Model, self).__init__()

        self.point_num = point_num
        self.input_channel = input_channel
        self.mid_channels = mid_channels

        self.data_bn = nn.BatchNorm1d(self.point_num * self.input_channel)

        Graph = import_class(graph)
        self.graph = Graph(**graph_arg)
        self.A = self.graph.A

        kernel_size = self.A.shape[0]
        t_kernel = 9

        self.backBone1 = MuiltKernelGTCN_(input_channel, 64, self.A, True, kernel_num, (t_kernel, kernel_size), edge_weight, lamda, dropout=dropout)
        self.backBone2 = MuiltKernelGTCN_(64, 64, self.A, True, kernel_num, (t_kernel, kernel_size), edge_weight, lamda, stride=1, dropout=dropout)
        self.backBone3 = MuiltKernelGTCN_(64, 64, self.A, True, kernel_num, (t_kernel, kernel_size), edge_weight, lamda, dropout=dropout)

        self.input_att = MultiHeadedGraphAttention(64, heads, mid_channels[0])
        self.input_layer = nn.ModuleList(
            [
                MultiHeadedGraphAttention(mid_channels[i], heads, mid_channels[i+1])
                for i in range(len(self.mid_channels)-1)
            ]
        )
        self.position_encoding = import_class(position_encoding)

        self.encoder = Encoder(layer_num=layer_num,
                                position_encoding=self.position_encoding, 
                                heads=heads, 
                                size=encode_size, 
                                d_ff=d_ff, 
                                dropout=dropout, 
                                attention_dropout=attention_dropout, 
                                position_encoding_dropout=position_encoding_dropout)

        self.average_pooling = nn.AdaptiveAvgPool2d((8,8))
        self.out_pooling = nn.AdaptiveAvgPool1d(20)
        self.output_layer = nn.Linear(20*8*8, n_classes)


    def forward(self, x):
        N, C, T, V, M = x.size()  # for NTU, (N, 3, T, 25, M)
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # for NTU, (N, M, 25, 3, T)
        x = x.view(N * M, V * C, T)  # for NTU, (N*M, 75, T)
        x = self.data_bn(x)
        x = x.view(N*M, V, C, T)  #for NTU,  (N*M,25,3,T)
        x = x.permute(0, 2, 3, 1).contiguous()  # for NTU, (NM, 3, T, 25)
        # x = x.view(N*M*T, V, C)


        x1 = self.backBone1(x)
        x2 = self.backBone2(x1)
        x3 = self.backBone3(x2)
        # print(x3.shape)

        _, C, T, V = x3.shape
        x = x3.permute(0, 2, 3, 1).contiguous()
        x = x.view(N*M*T, V, C)
        # print(x.shape)

        encoded_x = self.input_att(x, x, x)
        for layer in self.input_layer:
            encoded_x = layer(encoded_x, encoded_x, encoded_x)
        # print(encoded_x.shape)

        encoded_x = encoded_x.view(N, M, T, V, -1)
        encoded_x = encoded_x.permute(0, 1, 3, 2, 4).contiguous()
        encoded_x = encoded_x.view(N*M*V, T, -1)

        # print(encoded_x.shape)
        # print(encoded_x.shape)

        out = self.encoder(encoded_x)
        _, T, _ = out.shape
        # print(out.shape)

        out = out.view(N, M*V, T, -1).permute(0, 2, 1, 3).contiguous()
        # print(out.shape)
        out = self.average_pooling(out).squeeze_()
        out = F.leaky_relu(out)
        out = out.view(N, T,  -1).permute(0, 2, 1).contiguous()
        out = self.out_pooling(out).permute(0, 2, 1).contiguous()
        out = out.view(N, -1)
        out = self.output_layer(out)

        # print(out)
        return out






if __name__ == "__main__":
    m = Model(
        graph='graph.ntu_rgb_d.Graph',
        kernel_num=8,
        edge_weight=True,
        lamda=1,
        
        input_channel=3,
        mid_channels=[32, 256],
        layer_num=4,
        position_encoding='modules.position_encoding.PositionalEncoding', 
        heads=8, 
        encode_size=256, 
        d_ff=256, 
        dropout=0, 
        attention_dropout=0, 
        position_encoding_dropout=0,
        point_num=25, 
        n_classes=60
    ).to('cuda:0')
    a = torch.randn((8, 3, 300, 25, 2)).to('cuda:0')
    y = m(a)
    print(y.shape)


