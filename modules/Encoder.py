# # -*- encoding: utf-8 -*-
'''
@File    :   Encoder.py
@Time    :   2021/03/05 16:43:11
@Author  :   zqp 
@Version :   1.0
@Contact :   zhangqipeng@buaa.edu.cn
'''


import torch 
import torch.nn as nn
import sys
# sys.path.append('/home2/zqp/TR-AR')
from modules.multihead_stride_attention import MultiHeadedAttention
from modules.position_encoding import PositionalEncoding


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(self, input_size, ff_size, dropout=0.1):
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """
        super(PositionwiseFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.pwff_layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        return self.pwff_layer(x_norm) + x


class EncoderLayer(nn.Module):
    def __init__(self, heads, size, d_ff, dropout, attention_dropout, stride=2):
        super(EncoderLayer, self).__init__()

        self.heads = heads
        self.size = size
        self.d_ff = d_ff
        self.dropout = dropout
        self.stride = stride
        self. attention_dropout = attention_dropout

        self.attention = MultiHeadedAttention(self.heads, self.size, self.stride,  self.attention_dropout)
        self.feed_forward = PositionwiseFeedForward(self.size, self.d_ff, self.dropout)

        self.layer_norm = nn.LayerNorm(self.size, eps=1e-6)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, inputs, mask):
        """[summary]

        Args:
            input (tensor): bs*src_len*model_dim
            mask (tensor): (batch_size, 1, src_len)
        """
        input_norm = self.layer_norm(inputs)
        out = self.attention(input_norm, input_norm, input_norm, mask)

        out = self.dropout_layer(out)# TODO: 增加残差结构

        return self.feed_forward(out)


class Encoder(nn.Module):
    def __init__(self, layer_num, position_encoding, heads, size, d_ff, dropout, attention_dropout, position_encoding_dropout):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    heads = heads,
                    size=size,
                    d_ff=d_ff,
                    dropout=dropout,
                    attention_dropout=attention_dropout
                )
                for num in range(layer_num)
            ]
        )

        self.position_encoding = position_encoding(size)

        self.position_encoding_dropout = nn.Dropout(position_encoding_dropout)
        
        self.layer_num = nn.LayerNorm(size, eps=1e-6)

    def forward(self, inputs, mask=None):
        x = self.position_encoding(inputs)
        x = self.position_encoding_dropout(x)

        for layer in self.layers:
            x = layer(x, mask)
        
        out = self.layer_num(x)

        return out


if __name__ == "__main__":
    a = torch.rand((16, 300, 512)).to('cuda')

    encode = Encoder(layer_num=3,
                    position_encoding = PositionalEncoding,
                    heads=8, 
                    size=512, 
                    d_ff=512, 
                    dropout=0.1, 
                    attention_dropout=0.1, 
                    position_encoding_dropout=0.1).to('cuda')

    out = encode(a, None)
    print(out.shape)