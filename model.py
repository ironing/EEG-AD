import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional



def conv1x3(in_planes, out_planes, stride=1):
    "1x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3), stride=(1, stride),
                     padding=(0, 1), bias=False)


def conv1x7(in_planes, out_planes, stride=1):
    "1x7 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 7), stride=(1, stride),
                     padding=(0, 3), bias=False)


def conv3x1(in_planes, out_planes, stride=1):
    "3x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 1), stride=(stride, 1),
                     padding=(1, 0), bias=False)


def conv1x13(in_planes, out_planes, stride=3):
    "1x13 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 13), stride=(1, stride),
                     padding=(0, 3), bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)







class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.inplanes = inplanes
        self.stride = stride
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(self.inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv1x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * 4)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = conv1x3(self.inplanes, planes * 4, stride)

        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)
        identity = self.bn3(identity)

        out += identity
        out = self.relu(out)

        return out






class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=(1, 3), stride=(1, stride), padding=(0, 1), bias=False),
                nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out









class BasicBlock1(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock1, self).__init__()
        self.conv1 = conv1x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=(1, 3), stride=(1, stride), padding=(0, 1), bias=False),
                nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out









class BasicBlock2(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock2, self).__init__()
        self.conv1 = conv1x7(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x7(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=(1, 7), stride=(1, stride), padding=(0, 3), bias=False),
                nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print('out:', out.shape)

        out = self.conv2(out)
        out = self.bn2(out)
        # print('out:', out.shape)

        # print('res:', residual.shape)

        out += residual
        out = self.relu(out)

        return out






class ResNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock2]],
            inplane: int = 16,
            num_classes: int = 10,
    ) -> None:
        super(ResNet, self).__init__()
        self.inplane = inplane
        self.conv1 = nn.Conv2d(1, self.inplane, kernel_size=(1, 7), stride=(1, 2),
                               padding=(0, 3),
                               bias=False)

        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, self.inplane, blocks=3, stride=1)
        self.layer2 = self._make_layer(block, self.inplane * 2, blocks=4, stride=2)
        self.layer3 = self._make_layer(block, self.inplane * 2, blocks=6, stride=2)
        self.layer4 = self._make_layer(block, self.inplane * 2, blocks=3, stride=2)

       
        self.conv2 = nn.Conv2d(self.inplane, 16, kernel_size=(1, 7), stride=(1, 1),
                               padding=(0, 0),
                               bias=False)

        
        self.conv3 = nn.Conv2d(inplane, 2, kernel_size=(18, 7), stride=(1, 3),
                               padding=(0, 0),
                               bias=False)

        
        self.avgpool = nn.AdaptiveAvgPool2d((18, 1))

        self.fc = nn.Linear(542, num_classes)

        self.initialize()


    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride):

        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride))
            self.inplane = planes

        return nn.Sequential(*layers)


    def forward(self, x): 
        # print('input:', x.shape)
        x = self.conv1(x)   
        # print('0:', x.shape)

        x = self.bn1(x)       
        # print('1:', x.shape)

        x1 = self.relu(x)    
        # print('2:', x1.shape)

        x2 = self.layer1(x1)   
        # print('3:', x2.shape)

        x2 = self.layer2(x2)   
        # print('4:', x2.shape)

        x2 = self.layer3(x2)   
        # print('5:', x2.shape)

        x2 = self.layer4(x2)   
        # print('6:', x2.shape)

        x2 = self.conv2(x2)    
        # print('7:', x2.shape)

        x2 = self.avgpool(x2)  
        # print('8:', x2.shape)

        out1 = x2.view(x2.size(0), -1)    
        # print('9:', out1.shape)

        out2 = self.conv3(x1)             
        # print('10:', out2.shape)

        out2 = out2.view(out2.size(0), -1)    
        # print('11:', out2.shape)

        embeds = torch.cat((out1, out2), dim=-1)   
        # print('12:', embeds.shape)

        logits = self.fc(embeds)    # b x num_classes
        # print('----------------------------')

        return embeds, logits


      

def TaskOriNet(**kwargs):
    model = ResNet(BasicBlock2, **kwargs)
    return model
