import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet, BasicBlock
import numpy as np
import torchvision

from functools import partial

device = 'cuda' if torch.cuda.is_available() else 'cpu'



############################################################################################################
# MOBILENET
############################################################################################################
class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, norm_layer):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                norm_layer(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)

    def __init__(self, num_classes=10, norm_layer=nn.BatchNorm2d,shrink=1):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.norm_layer = norm_layer
        self.cfg = [(1,  16//shrink, 1, 1),
                   (6,  24//shrink, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
                   (6,  32//shrink, 3, 2),
                   (6,  64//shrink, 4, 2),
                   (6,  96//shrink, 3, 1),
                   (6, 160//shrink, 3, 2),
                   (6, 320//shrink, 1, 1)]


        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.norm_layer(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(self.cfg[-1][1], 1280//shrink, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = self.norm_layer(1280//shrink)


        self.classification_layer = nn.Linear(1280//shrink, num_classes)


    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride, self.norm_layer))
                in_planes = out_planes
        return nn.Sequential(*layers)


    def extract_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


    def forward(self, x):
        feature = self.extract_features(x)
        out = self.classification_layer(feature)
        return out




def mobilenetv2s(num_classes=10):
    return MobileNetV2(norm_layer=nn.BatchNorm2d, shrink=2, num_classes=num_classes)


    

############################################################################################################
# RESNET
############################################################################################################

class Model(nn.Module):
    def __init__(self, feature_dim=128, group_norm=False):
        super(Model, self).__init__()

        self.f = []
        for name, module in ResNet(BasicBlock, [1,1,1,1], num_classes=10).named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

        if group_norm:
            apply_gn(self)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class resnet8(nn.Module):
    def __init__(self, num_classes=10, pretrained_path=None, group_norm=False):
        super(resnet8, self).__init__()

        # encoder
        self.f = Model(group_norm=group_norm).f
        # classifier
        self.classification_layer = nn.Linear(512, num_classes, bias=True)


        if pretrained_path:
            self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)


    def extract_features(self, x):
        return torch.flatten(self.f(x), start_dim=1)


    def forward(self, x):
        feature = self.extract_features(x)
        out = self.classification_layer(feature)
        return out




############################################################################################################
# SHUFFLENET
############################################################################################################



class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).reshape(N,C,H,W)


class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups):
        super(Bottleneck, self).__init__()
        self.stride = stride

        mid_planes = out_planes//4
        g = 1 if in_planes==24 else groups
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.shuffle1 = ShuffleBlock(groups=g)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = self.shortcut(x)
        out = F.relu(torch.cat([out,res], 1)) if self.stride==2 else F.relu(out+res)
        return out


class ShuffleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ShuffleNet, self).__init__()
        cfg = {'out_planes': [200,400,800],'num_blocks': [4,8,4],'groups': 2}

        out_planes = cfg['out_planes']
        num_blocks = cfg['num_blocks']
        groups = cfg['groups']

        self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_planes = 24
        self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
        self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
        self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
        self.classification_layer = nn.Linear(out_planes[2], num_classes)
     

    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            cat_planes = self.in_planes if i == 0 else 0
            layers.append(Bottleneck(self.in_planes, out_planes-cat_planes, stride=stride, groups=groups))
            self.in_planes = out_planes
        return nn.Sequential(*layers)


    def extract_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        feature = out.view(out.size(0), -1)
        return feature


    def forward(self, x):
        feature = self.extract_features(x)
        out = self.classification_layer(feature)
        return out






def get_model(model):

  return  {   "mobilenetv2s" : (mobilenetv2s, optim.Adam, {"lr" : 0.001}),
              "shufflenet" : (ShuffleNet, optim.Adam, {"lr" : 0.001}),
                "resnet8" : (resnet8, optim.Adam, {"lr" : 0.001})
          }[model]


def print_model(model):
  n = 0
  print("Model:")
  for key, value in model.named_parameters():
    print(' -', '{:30}'.format(key), list(value.shape), "Requires Grad:", value.requires_grad)
    n += value.numel()
  print("Total number of Parameters: ", n) 
  print()






