# coding=utf-8
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
def conv3x3(in_channels,out_channels,stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )
def conv1x1(in_channels,out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        bias=False
    )
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self,in_channels,out_channels,stride=1,downsample=None):
        super(Bottleneck, self).__init__()
        self.outchannels = out_channels
        self.conv1 = conv1x1(in_channels,out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = conv3x3(out_channels,out_channels,stride)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = conv1x1(out_channels,out_channels*self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)

        self.conv4 = conv1x1(out_channels*self.expansion,2)
        self.relu = nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride

        self.sigmoid  = nn.Sigmoid()
    def mask(self,x,soft = False):
        if soft == True:
            mask = self.conv4(x)
            mask = torch.mean(mask)
            mask = mask.unsqueeze(1)
            mask = mask.repeat(1, self.outchannels * self.expansion, 1, 1)
            return mask
        '''
        representation = self.conv4(x)
        '''
        representation = torch.sum(x,dim=1)
        mask = self.sigmoid(representation)
        temp = mask.view(mask.size(0),-1)
        threshold = torch.mean(temp,dim=1)
        std = torch.std(temp,dim=1)
        threshold = threshold.unsqueeze(1)
        threshold = threshold.unsqueeze(1)
        std = std.unsqueeze(1)
        std = std.unsqueeze(1)
        mask = torch.where(mask<threshold+2*std,torch.ones_like(mask),torch.zeros_like(mask))
        mask = mask.unsqueeze(1)
        mask = mask.repeat(1,self.outchannels*self.expansion,1,1)
        print(mask.shape)
        return mask


    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)  # 目的同上
        #mask = self.mask(residual)
        out += residual#*mask
        out = self.relu(out)
        return out
class ResNet(nn.Module):
    def mask(self,x):
        representation = torch.mean(x,dim=1)
        representation = representation**2
        mask = self.sigmoid(representation)
        temp = mask.view(mask.size(0),-1)
        threshold = torch.mean(temp,dim=1)
        std = torch.std(temp,dim=1)
        threshold = threshold.unsqueeze(1)
        threshold = threshold.unsqueeze(1)
        std = std.unsqueeze(1)
        std = std.unsqueeze(1)
        mask = torch.where(mask<threshold+2*std,torch.ones_like(mask),torch.zeros_like(mask))
        mask = mask.unsqueeze(1)
        mask = mask.repeat(1,self.outchannels*self.expansion,1,1)
        print(mask.shape)
        return mask
    def _make_layer(self,block,out_channels,blocks,stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels,out_channels,stride,downsample))#先手动创建一个层，其余的层就是简单的添加
        #stride仅仅使用了如上一次，其余的都是标准的不变
        self.in_channels = out_channels * block.expansion
        for i in range(1, blocks):  # 这里面所有的block
            layers.append(block(self.in_channels, out_channels))
            # 一定要注意，out_channels一直都是3*3卷积层的深度
        return nn.Sequential(*layers)

    def __init__(self,block,layers,num_classes=10):
        super(ResNet,self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])  # 特征图大小不变
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 特征图缩小1/2
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 特征图缩小1/2
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 特征图缩小1/2
        self.avgpool = nn.AvgPool2d(1, stride=1)  # 平均池化，滤波器为7*7，步长为1，特征图大小变为1*1
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 全连接层

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        mask = self.mask(x)
        x = x*mask
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)  # 写代码时一定要仔细，别把out写成x了，我在这里吃了好大的亏
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)  # 将原有的多维输出拉回一维
        out = self.fc(out)
        return out
def resnet101():
    return ResNet(Bottleneck,[3,4,23,3])
def resnet18():
    return ResNet(Bottleneck, [2,2, 2,2])
test_net = resnet101()
test_x = Variable(torch.rand(2,3,32,32))
test_y = test_net(test_x)
print(test_y.shape)