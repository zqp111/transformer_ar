import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride=1)
        self.bn1 = self.norm_layer(planes, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # downsample
        self.conv2 = conv3x3(planes, planes, stride=1)
        self.bn2 = self.norm_layer(planes, affine=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # print("==== ", x.shape)
        return x


class CnnBacknone(nn.Module):
    def __init__(self):
        super(CnnBacknone, self).__init__()

        self.conv = conv3x3(4, 32)
        self.bn = nn.BatchNorm2d(32, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

        # 4 basic blocks
        channels = [32, 64, 128, 256, 512]
        layers = []
        for num_layer in range(len(channels) - 1):
            layers.append(BasicBlock(channels[num_layer], channels[num_layer + 1]))
        self.layers = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.cnn_bn = nn.BatchNorm1d(512)

    def forward(self, x):
        """
        :param x: [bs, T, 4, 24, 72]
        :return:
        """
        bs, T, c, lon, lat = x.size()

        x = x.reshape(-1, c, lon, lat)

        x = self.conv(x)
        #print("1. ", x.shape)
        x = self.bn(x)
        x = self.relu(x)
        # x = self.pool(x)
        #print("2. ", x.shape)
        x = self.layers(x)
        #print("3. ", x.shape)
        x = self.avgpool(x).squeeze_()  # [bs*t, 512]
        #print("4. ", x.shape)
        x = self.cnn_bn(x)
        x = self.relu(x)
        x = x.reshape(bs, -1, 512)     # [bs, t ,512]
        return x


if __name__ == "__main__":
    x = torch.randn(16, 12, 4, 24, 72)
    model = CnnBacknone()
    out = model(x)
    #print(out.shape)
