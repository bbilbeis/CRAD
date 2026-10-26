import torch
import torch.nn as nn
import math
from torch.autograd import Function

class ShakeDropFunction(Function):
    @staticmethod
    def forward(ctx, x, training=True, p_drop=0.5, alpha_range=[-1, 1]):
        if int(training) == 1:
            gate = torch.cuda.FloatTensor([0]).bernoulli_(1 - p_drop)
            ctx.save_for_backward(gate)
            if gate.item() == 0:
                alpha = torch.cuda.FloatTensor(x.size(0)).uniform_(*alpha_range)
                alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x)
                return alpha * x
            else:
                return x
        else:
            return (1 - p_drop) * x

    @staticmethod
    def backward(ctx, grad_output):
        gate = ctx.saved_tensors[0]
        if gate.item() == 0:
            beta = torch.cuda.FloatTensor(grad_output.size(0)).uniform_(0, 1)
            beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad_output)
            beta = torch.autograd.Variable(beta)
            return beta * grad_output, None, None, None
        else:
            return grad_output, None, None, None

class ShakeDrop(nn.Module):
    def __init__(self, p_drop=0.5, alpha_range=[-1, 1]):
        super(ShakeDrop, self).__init__()
        self.p_drop = p_drop
        self.alpha_range = alpha_range

    def forward(self, x):
        return ShakeDropFunction.apply(x, self.training, self.p_drop, self.alpha_range)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    outchannel_ratio = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, p_drop=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.shake_drop = ShakeDrop(p_drop) if p_drop > 0 else nn.Identity()

    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        
        out = self.shake_drop(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(torch.zeros(batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1]).to(x.device))
            out = out + torch.cat((shortcut, padding), 1)
        else:
            out = out + shortcut

        return out

class Bottleneck(nn.Module):
    outchannel_ratio = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, p_drop=0.0):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, (planes*1), kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d((planes*1))
        self.conv3 = nn.Conv2d((planes*1), planes * Bottleneck.outchannel_ratio, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes * Bottleneck.outchannel_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.shake_drop = ShakeDrop(p_drop) if p_drop > 0 else nn.Identity()

    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn4(out)
        
        out = self.shake_drop(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(torch.zeros(batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1]).to(x.device))
            out = out + torch.cat((shortcut, padding), 1)
        else:
            out = out + shortcut

        return out


class PyramidNet(nn.Module):
    def __init__(self, depth, alpha, num_classes, bottleneck=True):
        super(PyramidNet, self).__init__()
        self.inplanes = 16
        if bottleneck:
            n = int((depth - 2) / 9)
            block = Bottleneck
        else:
            n = int((depth - 2) / 6)
            block = BasicBlock

        self.addrate = alpha / (3*n*1.0)
        self.input_featuremap_dim = self.inplanes
        self.conv1 = nn.Conv2d(3, self.input_featuremap_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.input_featuremap_dim)

        self.featuremap_dim = self.input_featuremap_dim 
        self.layer1 = self.pyramidal_make_layer(block, n)
        self.layer2 = self.pyramidal_make_layer(block, n, stride=2)
        self.layer3 = self.pyramidal_make_layer(block, n, stride=2)

        self.final_featuremap_dim = self.input_featuremap_dim
        self.bn_final= nn.BatchNorm2d(self.final_featuremap_dim)
        self.relu_final = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.final_featuremap_dim, num_classes)
        
        self.u_idx = 0
        self.total_u = 3 * n
        self.p_drop_base = 0.5
        
        # apply drop probability linearly increasing across depth
        for m in self.modules():
            if isinstance(m, ShakeDrop):
                self.u_idx += 1
                m.p_drop = (self.u_idx / self.total_u) * self.p_drop_base

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n_weight = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n_weight))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def pyramidal_make_layer(self, block, block_depth, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d((2,2), stride=(2, 2), ceil_mode=True)

        layers = []
        self.featuremap_dim = self.featuremap_dim + self.addrate
        layers.append(block(self.input_featuremap_dim, int(round(self.featuremap_dim)), stride, downsample, p_drop=0.5))
        for i in range(1, block_depth):
            temp_featuremap_dim = self.featuremap_dim + self.addrate
            layers.append(block(int(round(self.featuremap_dim)) * block.outchannel_ratio, int(round(temp_featuremap_dim)), 1, p_drop=0.5))
            self.featuremap_dim  = temp_featuremap_dim
        self.input_featuremap_dim = int(round(self.featuremap_dim)) * block.outchannel_ratio

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn_final(x)
        x = self.relu_final(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
