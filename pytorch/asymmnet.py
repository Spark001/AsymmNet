__all__ = ['AsymmNet_Large', 'AsymmNet_Small', 'AsymmNet']

from torch import nn
from torch import cat
import math
import torch.nn.functional as F


class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class HardSigmoid(nn.Module):
    def __init__(self, inplace=False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class Activation(nn.Module):
    def __init__(self, act_func):
        super(Activation, self).__init__()
        if act_func == "relu":
            self.act = nn.ReLU()
        elif act_func == "relu6":
            self.act = nn.ReLU6()
        elif act_func == "hard_sigmoid":
            self.act = HardSigmoid()
        elif act_func == "hard_swish":
            self.act = HardSwish()
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.act(x)


def make_divisible(x, divisible_by=8):
    return int(math.ceil(x * 1. / divisible_by) * divisible_by)


class _BasicUnit(nn.Module):
    def __init__(self, num_in, num_out, kernel_size=1, strides=1, pad=0, num_groups=1,
                 use_act=True, act_type="relu", norm_layer=nn.BatchNorm2d):
        super(_BasicUnit, self).__init__()
        self.use_act = use_act
        self.conv = nn.Conv2d(in_channels=num_in, out_channels=num_out,
                              kernel_size=kernel_size, stride=strides,
                              padding=pad, groups=num_groups, bias=False,
                              )
        self.bn = norm_layer(num_out)
        if use_act is True:
            self.act = Activation(act_type)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        if self.use_act:
            out = self.act(out)
        return out


class SE_Module(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SE_Module, self).__init__()
        reduction_c = make_divisible(channels // reduction)
        self.out = nn.Sequential(
            nn.Conv2d(channels, reduction_c, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction_c, channels, 1, bias=True),
            HardSigmoid()
        )

    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, 1)
        y = self.out(y)
        return x * y


class AsymmBottleneck(nn.Module):
    def __init__(self, num_in, num_mid, num_out, kernel_size, asymmrate=1,
                 act_type="relu", use_se=False, strides=1,
                 norm_layer=nn.BatchNorm2d):
        super(AsymmBottleneck, self).__init__()
        assert isinstance(asymmrate, int)
        self.asymmrate = asymmrate
        self.use_se = use_se
        self.use_short_cut_conv = (num_in == num_out and strides == 1)
        self.do_expand = (num_mid > max(num_in, asymmrate * num_in))
        if self.do_expand:
            self.expand = _BasicUnit(num_in, num_mid - asymmrate * num_in,
                                     kernel_size=1,
                                     strides=1, pad=0, act_type=act_type,
                                     norm_layer=norm_layer)
            num_mid += asymmrate * num_in
        self.dw_conv = _BasicUnit(num_mid, num_mid, kernel_size, strides,
                                  pad=self._get_pad(kernel_size), act_type=act_type,
                                  num_groups=num_mid, norm_layer=norm_layer)
        if self.use_se:
            self.se = SE_Module(num_mid)
        self.pw_conv_linear = _BasicUnit(num_mid, num_out, kernel_size=1, strides=1,
                                         pad=0, act_type=act_type, use_act=False,
                                         norm_layer=norm_layer, num_groups=1)

    def forward(self, x):
        if self.do_expand:
            out = self.expand(x)
            feat = []
            for i in range(self.asymmrate):
                feat.append(x)
            feat.append(out)
            for i in range(self.asymmrate):
                feat.append(x)
            if self.asymmrate > 0:
                out = cat(feat, dim=1)
        else:
            out = x
        out = self.dw_conv(out)
        if self.use_se:
            out = self.se(out)
        out = self.pw_conv_linear(out)
        if self.use_short_cut_conv:
            return x + out
        return out

    def _get_pad(self, kernel_size):
        if kernel_size == 1:
            return 0
        elif kernel_size == 3:
            return 1
        elif kernel_size == 5:
            return 2
        elif kernel_size == 7:
            return 3
        else:
            raise NotImplementedError


def get_asymmnet_cfgs(model_name):
    if model_name == 'asymmnet_large':
        inplanes = 16
        cfg = [
            # k, exp, c,  se,     nl,  s,
            # stage1
            [3, 16, 16, False, 'relu', 1],
            # stage2
            [3, 64, 24, False, 'relu', 2],
            [3, 72, 24, False, 'relu', 1],
            # stage3
            [5, 72, 40, True, 'relu', 2],
            [5, 120, 40, True, 'relu', 1],
            [5, 120, 40, True, 'relu', 1],
            # stage4
            [3, 240, 80, False, 'hard_swish', 2],
            [3, 200, 80, False, 'hard_swish', 1],
            [3, 184, 80, False, 'hard_swish', 1],
            [3, 184, 80, False, 'hard_swish', 1],
            [3, 480, 112, True, 'hard_swish', 1],
            [3, 672, 112, True, 'hard_swish', 1],
            # stage5
            [5, 672, 160, True, 'hard_swish', 2],
            [5, 960, 160, True, 'hard_swish', 1],
            [5, 960, 160, True, 'hard_swish', 1],
        ]
        cls_ch_squeeze = 960
        cls_ch_expand = 1280
    elif model_name == 'asymmnet_small':
        inplanes = 16
        cfg = [
            # k, exp, c,  se,     nl,  s,
            [3, 16, 16, True, 'relu', 2],
            [3, 72, 24, False, 'relu', 2],
            [3, 88, 24, False, 'relu', 1],
            [5, 96, 40, True, 'hard_swish', 2],
            [5, 240, 40, True, 'hard_swish', 1],
            [5, 240, 40, True, 'hard_swish', 1],
            [5, 120, 48, True, 'hard_swish', 1],
            [5, 144, 48, True, 'hard_swish', 1],
            [5, 288, 96, True, 'hard_swish', 2],
            [5, 576, 96, True, 'hard_swish', 1],
            [5, 576, 96, True, 'hard_swish', 1],
        ]
        cls_ch_squeeze = 576
        cls_ch_expand = 1280
    else:
        raise ValueError('{} model_name is not supported now!'.format(model_name))

    return inplanes, cfg, cls_ch_squeeze, cls_ch_expand


class AsymmNet(nn.Module):
    def __init__(self, cfgs_name, num_classes=1000, multiplier=1.0, asymmrate=1, dropout_rate=0.2,
                 norm_layer=nn.BatchNorm2d):
        super(AsymmNet, self).__init__()
        inplanes, cfg, cls_ch_squeeze, cls_ch_expand = get_asymmnet_cfgs(cfgs_name)
        k = multiplier
        self.inplanes = make_divisible(inplanes * k)
        self.first_block = nn.Sequential(
            nn.Conv2d(3, self.inplanes, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.inplanes),
            HardSwish(inplace=True),
        )

        asymm_layers = []
        for layer_cfg in cfg:
            layer = self._make_layer(kernel_size=layer_cfg[0],
                                     exp_ch=make_divisible(k * layer_cfg[1]),
                                     out_channel=make_divisible(k * layer_cfg[2]),
                                     use_se=layer_cfg[3],
                                     act_func=layer_cfg[4],
                                     asymmrate=asymmrate,
                                     stride=layer_cfg[5],
                                     norm_layer=norm_layer,
                                     )
            asymm_layers.append(layer)
        self.asymm_block = nn.Sequential(*asymm_layers)
        self.last_block = nn.Sequential(
            nn.Conv2d(self.inplanes, make_divisible(k * cls_ch_squeeze), 1, bias=False),
            nn.BatchNorm2d(make_divisible(k * cls_ch_squeeze)),
            HardSwish(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(make_divisible(k * cls_ch_squeeze), cls_ch_expand, 1, bias=False),
            HardSwish(),
            nn.Dropout2d(p=dropout_rate, inplace=True),
            nn.Flatten(),
        )
        self.output = nn.Linear(cls_ch_expand, num_classes)

    def _make_layer(self, kernel_size, exp_ch, out_channel, use_se, act_func, asymmrate, stride,
                    norm_layer):
        mid_planes = exp_ch
        out_planes = out_channel
        layer = AsymmBottleneck(self.inplanes, mid_planes,
                                out_planes, kernel_size, asymmrate,
                                act_func, strides=stride, use_se=use_se, norm_layer=norm_layer)
        self.inplanes = out_planes
        return layer

    def forward(self, x):
        x = self.first_block(x)
        x = self.asymm_block(x)
        x = self.last_block(x)
        x = self.output(x)
        return x


class AsymmNet_Large(AsymmNet):
    def __init__(self, **kwargs):
        super(AsymmNet_Large, self).__init__(cfgs_name='asymmnet_large', **kwargs)


class AsymmNet_Small(AsymmNet):
    def __init__(self, **kwargs):
        super(AsymmNet_Small, self).__init__(cfgs_name='asymmnet_small', **kwargs)


if __name__ == '__main__':
    from torchtoolbox.tools.summary import summary
    import torch

    x = torch.rand((1, 3, 224, 224))
    # summary(AsymmNet_Small(), x)
    summary(AsymmNet_Large(multiplier=1.0), x)
    # model = AsymmNet(cfgs_name='asymmnet_large')
    # summary(model, x)
