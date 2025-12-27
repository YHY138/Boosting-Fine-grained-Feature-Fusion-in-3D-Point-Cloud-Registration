import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch

# __all__ = ['Res2Net', 'res2net50', 'Bottle2neck', 'my_res2Net', 'my_Bottle2neck']

model_urls = {
    'res2net50_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_4s-06e79181.pth',
    'res2net50_48w_2s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_48w_2s-afed724a.pth',
    'res2net50_14w_8s': './res2net50_14w_8s-6527dddc.pth',
    'res2net50_26w_6s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_6s-19041792.pth',
    'res2net50_26w_8s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_8s-2c7c9f12.pth',
    'res2net101_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_26w_4s-02a759a1.pth',
}


class Bottle2neck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3，应该就是论文中的group c
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()
# 这里的64应该是论文在实验部分展示的ResNet-50的width，ResNet-50的scale为 1.所以后续想用其他width、scale需要以64为基准？？
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        # 这里的convs对应论文中的每个 3x3 的groupC卷积。然后还会加上一个BatchNorm归一化模块
        # 这里就刚好把上一个卷积层的输出维度平均分割成了nums个子维度（nums和scale相关），每个子维度的通道数是width
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class my_Bottle2neck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3，应该就是论文中的group c
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(my_Bottle2neck, self).__init__()
# 这里的64应该是论文在实验部分展示的ResNet-50的width，ResNet-50的scale为 1.所以后续想用其他width、scale需要以64为基准？？
        width = int(math.floor(planes * (baseWidth / 64.0)))

        self.conv1 = nn.Linear(inplanes, width*scale, bias=False)
        self.bn1 = nn.BatchNorm1d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool1d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        # 这里的convs对应论文中的每个 3x3 的groupC卷积。然后还会加上一个BatchNorm归一化模块
        # 这里就刚好把上一个卷积层的输出维度平均分割成了nums个子维度（nums和scale相关），每个子维度的通道数是width
        for i in range(self.nums):
            convs.append(nn.Linear(width, width, bias=False))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Linear(width*scale, planes, bias=False)
        self.bn3 = nn.BatchNorm1d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

# 根据scale获得等比分割数据量的大小width，用split将out按照width等间距的分割成scale个特征块
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# 这里是在Res2Net输出时使用一次dropout来降维输出的Res2Net
class my_Bottle2neck_dropout(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3，应该就是论文中的group c
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(my_Bottle2neck_dropout, self).__init__()
# 这里的64应该是论文在实验部分展示的ResNet-50的width，ResNet-50的scale为 1.所以后续想用其他width、scale需要以64为基准？？
        width = int(math.floor(planes * (baseWidth / 64.0)))

        self.conv1 = nn.Linear(inplanes, width*scale, bias=False)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.dropout1 = nn.Dropout(p=0.2)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool1d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        # 这里的convs对应论文中的每个 3x3 的groupC卷积。然后还会加上一个BatchNorm归一化模块
        # 这里就刚好把上一个卷积层的输出维度平均分割成了nums个子维度（nums和scale相关），每个子维度的通道数是width
        for i in range(self.nums):
            convs.append(nn.Linear(width, width, bias=False))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Linear(width*scale, planes, bias=False)
        self.bn3 = nn.BatchNorm1d(planes)

        # 下面两个都是自己加的
        # 加一个relu 测试版
        # self.relu3 = nn.ReLU(inplace=True)
        # 使用dropout防止过拟合 测试版
        # self.dropout3 = nn.Dropout(p=0.2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.dropout1(out) # 在上采样的时候加一个dropout
        out = self.bn1(out)
        out = self.relu(out)

# 根据scale获得等比分割数据量的大小width，用split将out按照width等间距的分割成scale个特征块
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)
        # 下面两句都是自己加的，看dropout能否防止过拟合。在下采样加dropout对精度有负面影响
        # out = self.relu3(out)
        # out = self.dropout3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class my_Bottle2neck2(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3，应该就是论文中的group c
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(my_Bottle2neck2, self).__init__()
# 这里的64应该是论文在实验部分展示的ResNet-50的width，ResNet-50的scale为 1.所以后续想用其他width、scale需要以64为基准？？
        width = int(math.floor(planes * (baseWidth / 64.0)))

        self.conv1 = nn.Linear(inplanes, width*scale, bias=False)
        self.bn1 = nn.BatchNorm1d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool1d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        # 这里的convs对应论文中的每个 3x3 的groupC卷积。然后还会加上一个BatchNorm归一化模块
        # 这里就刚好把上一个卷积层的输出维度平均分割成了nums个子维度（nums和scale相关），每个子维度的通道数是width
        for i in range(self.nums):
            convs.append(nn.Linear(width, width, bias=False))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Linear(width*scale, planes, bias=False)
        self.bn3 = nn.BatchNorm1d(planes)

        # self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.LeakyReLU(0.1)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Res2Net(nn.Module):
# layer参数是个list，每个元素表示的是每层由多少个Res2Net模块组成
    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64 # 这个inplanes应该和layerx前面部分网络的输出维度一致？？
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        # 先是一个3--64的卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 从这里开始写Bottle2neck部分
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 这里是全连接层对图片进行类别判断
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        print('(((((((((((( ori x.shape {} ))))))))))))'.format(x.shape))
        x = self.conv1(x)
        print('(((((((((((( conv1 x.shape {} ))))))))))))'.format(x.shape))
        x = self.bn1(x)
        print('(((((((((((( bn1 x.shape {} ))))))))))))'.format(x.shape))
        x = self.relu(x)
        print('(((((((((((( relu x shape {} )))))))))))))'.format(x.shape))
        x = self.maxpool(x)
        print("(((((((((((( maxpool x shape {} )))))))))))))".format(x.shape))
        x = self.layer1(x)
        print('(((((((((((( layer1 x shape {} )))))))))))))'.format(x.shape))
        x = self.layer2(x)
        print('(((((((((((( layer2 x shape {} )))))))))))))'.format(x.shape))
        x = self.layer3(x)
        print('(((((((((((( layer3 x shape {} )))))))))))))'.format(x.shape))
        x = self.layer4(x)
        print('(((((((((((( layer4 x shape {} )))))))))))))'.format(x.shape))
        x = self.avgpool(x)
        print('(((((((((((( avgpool x shape {} )))))))))))))'.format(x.shape))
        x = x.view(x.size(0), -1)
        print('(((((((((((( view x shape {} )))))))))))))'.format(x.shape))
        x = self.fc(x)
        print('(((((((((((( fc x shape {} )))))))))))))'.format(x.shape))

        return x

# 目前用这个来跑网络是效果比较好的
class my_res2Net(nn.Module):
    # layer参数是个list，每个元素表示的是每层由多少个Res2Net模块组成
    def __init__(self, block, in_dim, out_dim, baseWidth=26, scale=4):
        self.inplanes = in_dim
        self.out_dim = out_dim
        super(my_res2Net, self).__init__()
        # print('(((((((((( baseWidth ))))))))) {}'.format(baseWidth))
        # print("(((((((((( scale )))))))))) {}".format(scale))
        self.baseWidth = baseWidth
        self.scale = scale

        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 从这里开始写Bottle2neck部分
        self.layer1 = self._make_layer(block, self.out_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, outplanes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != outplanes * block.expansion:
            downsample = nn.Sequential(
                nn.Linear(self.inplanes, self.out_dim, bias=False),
                nn.BatchNorm1d(self.out_dim),
            )

        layers = []
        layers.append(block(self.inplanes, outplanes, stride, downsample=downsample,
                            stype='normal', baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print('(((((((((((( x shape {} )))))))))))))'.format(x.shape))
        x = self.layer1(x)

        return x

# 判断是否要在Res2Net最后一次卷积层后加上Dropout层
class my_res2Net_dropout(nn.Module):
    # layer参数是个list，每个元素表示的是每层由多少个Res2Net模块组成
    def __init__(self, block, in_dim, out_dim, baseWidth=26, scale=4):
        self.inplanes = in_dim
        self.out_dim = out_dim
        super(my_res2Net_dropout, self).__init__()
        # print('(((((((((( baseWidth ))))))))) {}'.format(baseWidth))
        # print("(((((((((( scale )))))))))) {}".format(scale))
        self.baseWidth = baseWidth
        self.scale = scale

        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 从这里开始写Bottle2neck部分
        self.layer1 = self._make_layer(block, self.out_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, outplanes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != outplanes * block.expansion:
            downsample = nn.Sequential(
                nn.Linear(self.inplanes, self.out_dim, bias=False),
                nn.BatchNorm1d(self.out_dim),
            )

        layers = []
        layers.append(block(self.inplanes, outplanes, stride, downsample=downsample,
                            stype='normal', baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print('(((((((((((( x shape {} )))))))))))))'.format(x.shape))
        x = self.layer1(x)

        return x

# 用这个来当作测试炼丹
class my_res2Net2(nn.Module):
    # layer参数是个list，每个元素表示的是每层由多少个Res2Net模块组成
    def __init__(self, block, in_dim, out_dim, baseWidth=26, scale=4):
        self.inplanes = in_dim
        self.out_dim = out_dim
        super(my_res2Net2, self).__init__()
        # print('(((((((((( baseWidth ))))))))) {}'.format(baseWidth))
        # print("(((((((((( scale )))))))))) {}".format(scale))
        self.baseWidth = baseWidth
        self.scale = scale

        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 从这里开始写Bottle2neck部分
        self.layer1 = self._make_layer(block, self.out_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, outplanes, stride=1):
        downsample = None
        # if stride != 1 or self.inplanes != outplanes * block.expansion:
        #     downsample = nn.Sequential(
        #         nn.Linear(self.inplanes, self.out_dim, bias=False),
        #         nn.BatchNorm1d(self.out_dim),
        #     )

        layers = []
        layers.append(block(self.inplanes, outplanes, stride, downsample=downsample,
                            stype='normal', baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print('(((((((((((( x shape {} )))))))))))))'.format(x.shape))
        x = self.layer1(x)

        return x


def res2net50(pretrained=False, **kwargs):
    """Constructs a Res2Net-50 model.
    Res2Net-50 refers to the Res2Net-50_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_4s']))
    return model


def res2net50_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_4s']))
    return model


def res2net101_26w_4s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net101_26w_4s']))
    return model


def res2net50_26w_6s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=6, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_6s']))
    return model


def res2net50_26w_8s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=8, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_26w_8s']))
    return model


def res2net50_48w_2s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_48w_2s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=48, scale=2, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_48w_2s']))
    return model


def res2net50_14w_8s(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_14w_8s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=14, scale=8, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load('./res2net50_14w_8s-6527dddc.pth'))
    return model

def test_models(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_14w_8s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [1, 1, 1, 1], baseWidth=14, scale=8, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load('./res2net50_14w_8s-6527dddc.pth'))
    return model


if __name__ == '__main__':
    images = torch.rand(1, 3, 224, 224).cuda(0)
    model = res2net50_14w_8s(pretrained=True).cuda(0)
    # model = test_models()
    # model = model.cuda(0)
    # print(model._modules)
    model(images)
    # print(model(images).size())