from __future__ import division

import numpy as np
import math
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.block import HybridBlock
from mxnet.context import cpu


class ReLU6(HybridBlock):
    """RelU6 used in MobileNetV2 and MobileNetV3.

    Parameters
    ----------
    kwargs : arguments goes to mxnet.gluon.nn.ReLU6
    """

    def __init__(self, **kwargs):
        super(ReLU6, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x, 0, 6, name="relu6")


class HardSigmoid(HybridBlock):
    """HardSigmoid used in MobileNetV3.

    Parameters
    ----------
    kwargs : arguments goes to mxnet.gluon.nn.HardSigmoid
    """

    def __init__(self, **kwargs):
        super(HardSigmoid, self).__init__(**kwargs)
        self.act = ReLU6()

    def hybrid_forward(self, F, x):
        return self.act(x + 3.) / 6.


class HardSwish(HybridBlock):
    """HardSwish used in MobileNetV3.

    Parameters
    ----------
    kwargs : arguments goes to mxnet.gluon.nn.HardSwish
    """

    def __init__(self, **kwargs):
        super(HardSwish, self).__init__(**kwargs)
        self.act = HardSigmoid()

    def hybrid_forward(self, F, x):
        return x * self.act(x)


class Activation(HybridBlock):
    """Activation function used in MobileNetV3"""

    def __init__(self, act_func, **kwargs):
        super(Activation, self).__init__(**kwargs)
        if act_func == "relu":
            self.act = nn.Activation('relu')
        elif act_func == "relu6":
            self.act = ReLU6()
        elif act_func == "hard_sigmoid":
            self.act = HardSigmoid()
        elif act_func == "swish":
            self.act = nn.Swish()
        elif act_func == "hard_swish":
            self.act = HardSwish()
        elif act_func == "leaky":
            self.act = nn.LeakyReLU(alpha=0.375)
        else:
            raise NotImplementedError

    def hybrid_forward(self, F, x):
        return self.act(x)


def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class _SE(HybridBlock):
    def __init__(self, num_out, ratio=4,
                 act_func=("relu", "hard_sigmoid"), use_bn=False, prefix='', **kwargs):
        super(_SE, self).__init__(**kwargs)
        self.use_bn = use_bn
        num_mid = make_divisible(num_out // ratio)
        self.pool = nn.GlobalAvgPool2D()
        self.conv1 = nn.Conv2D(channels=num_mid,
                               kernel_size=1, use_bias=True, prefix=('%s_fc1_' % prefix))
        self.act1 = Activation(act_func[0])
        self.conv2 = nn.Conv2D(channels=num_out,
                               kernel_size=1, use_bias=True, prefix=('%s_fc2_' % prefix))
        self.act2 = Activation(act_func[1])

    def hybrid_forward(self, F, x):
        out = self.pool(x)
        out = self.conv1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        return F.broadcast_mul(x, out)


class _Unit(HybridBlock):
    def __init__(self, num_out, kernel_size=1, strides=1, pad=0, num_groups=1,
                 use_act=True, act_type="relu", prefix='', norm_layer=BatchNorm, **kwargs):
        super(_Unit, self).__init__(**kwargs)
        self.use_act = use_act
        self.conv = nn.Conv2D(channels=num_out,
                              kernel_size=kernel_size, strides=strides,
                              padding=pad, groups=num_groups, use_bias=False,
                              prefix='%s-conv2d_' % prefix)
        self.bn = norm_layer(prefix='%s-batchnorm_' % prefix)
        if use_act is True:
            self.act = Activation(act_type)

    def hybrid_forward(self, F, x, *args):
        out = self.conv(x)
        out = self.bn(out)
        if self.use_act:
            out = self.act(out)
        return out


class _ChannelShuffle(HybridBlock):
    """
    ShuffleNet channel shuffle Block.
    """

    def __init__(self, channel, groups=2, **kwargs):
        super(_ChannelShuffle, self).__init__(**kwargs)
        assert channel % groups == 0
        self.groups = groups

    def hybrid_forward(self, F, x):
        data = F.reshape(x, shape=(0, -4, self.groups, -1, -2))  # [n,c,h,w] -> [n, g, c/g, h, w]
        data = F.swapaxes(data, 1, 2)  # [n, g, c/g, h, w] -> [n, c/g, g, h, w]
        data = F.reshape(data, shape=(0, -3, -2))  # [n, c/g, g, h, w] -> [n, c, h, w]
        return data


class AsymmBottleneck(HybridBlock):
    def __init__(self, num_in, num_mid, num_out, kernel_size, asymmrate=1,
                 act_type="relu", shuffle_group=0, use_se=False, strides=1,
                 prefix='', norm_layer=BatchNorm, **kwargs):
        super(AsymmBottleneck, self).__init__(**kwargs)
        assert isinstance(asymmrate, int)
        with self.name_scope():
            self.asymmrate = asymmrate
            self.use_se = use_se
            self.do_shuffle = (shuffle_group > 1 and asymmrate > 0)
            self.use_short_cut_conv = (num_in == num_out and strides == 1)
            self.do_expand = (num_mid > max(num_in, asymmrate * num_in))
            if self.do_expand:
                self.expand = _Unit(num_mid - asymmrate * num_in, kernel_size=1,
                                    strides=1, pad=0, act_type=act_type,
                                    prefix='%s-asym%dexp' % (prefix, asymmrate), norm_layer=norm_layer)
                num_mid += asymmrate * num_in
                if self.do_shuffle:
                    self.shuffle = _ChannelShuffle(num_mid, groups=shuffle_group)
            self.dw_conv = _Unit(num_mid, kernel_size=kernel_size, strides=strides,
                                 pad=self._get_pad(kernel_size),
                                 act_type=act_type, num_groups=num_mid,
                                 prefix='%s-depthwise' % prefix, norm_layer=norm_layer)
            if self.use_se:
                self.se = _SE(num_mid, prefix='%s-se' % prefix)
            self.pw_conv_linear = _Unit(num_out, kernel_size=1,
                                        strides=1, pad=0,
                                        act_type=act_type, use_act=False,
                                        prefix='%s-pwlinear' % prefix, norm_layer=norm_layer, num_groups=1)

    def hybrid_forward(self, F, x):
        if self.do_expand:
            out = self.expand(x)
            if self.asymmrate > 0:
                feat = []
                for i in range(self.asymmrate):
                    feat.append(x)
                feat.append(out)
                for i in range(self.asymmrate):
                    feat.append(x)
                out = F.concat(*feat, dim=1)
            if self.do_shuffle:
                out = self.shuffle(out)
        else:
            out = x
        out = self.dw_conv(out)
        if self.use_se:
            out = self.se(out)
        out = self.pw_conv_linear(out)
        if self.use_short_cut_conv:
            return x + out
        else:
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


class AsymmNet(HybridBlock):
    def __init__(self, inplanes, cfg, cls_ch_squeeze, cls_ch_expand, multiplier=1.,
                 classes=1000, norm_kwargs=None, last_gamma=False, asymmrate=1,
                 final_drop=0., use_global_stats=False, name_prefix='',
                 norm_layer=BatchNorm, **kwargs):
        super(AsymmNet, self).__init__(prefix=name_prefix)
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        if use_global_stats:
            norm_kwargs['use_global_stats'] = True
        # initialize residual networks
        k = multiplier
        self.last_gamma = last_gamma
        self.norm_kwargs = norm_kwargs
        self.inplanes = inplanes
        self.norm_layer = norm_layer

        with self.name_scope():
            self.features = nn.HybridSequential()
            self.features.add(nn.Conv2D(channels=make_divisible(k * self.inplanes), \
                                        kernel_size=3, padding=1, strides=2,
                                        use_bias=False, prefix='first-3x3-conv-conv2d_'))
            self.features.add(norm_layer(prefix='first-3x3-conv-batchnorm_'))
            self.features.add(HardSwish())
            i = 0
            for layer_cfg in cfg:
                layer = self._make_layer(kernel_size=layer_cfg[0],
                                         exp_ch=make_divisible(k * layer_cfg[1]),
                                         out_channel=make_divisible(k * layer_cfg[2]),
                                         use_se=layer_cfg[3],
                                         act_func=layer_cfg[4],
                                         stride=layer_cfg[5],
                                         asymmrate=asymmrate,
                                         prefix='seq-%d' % i,
                                         **kwargs
                                         )
                self.features.add(layer)
                i += 1
            if cls_ch_squeeze != 0:
                self.features.add(nn.Conv2D(channels=make_divisible(k * cls_ch_squeeze),
                                            kernel_size=1, padding=0, strides=1,
                                            use_bias=False, prefix='last-1x1-conv1-conv2d_'))
                self.features.add(norm_layer(prefix='last-1x1-conv1-batchnorm_',
                                             **({} if norm_kwargs is None else norm_kwargs)))
                self.features.add(HardSwish())
            self.features.add(nn.GlobalAvgPool2D())
            self.features.add(nn.Conv2D(channels=cls_ch_expand, kernel_size=1, padding=0, strides=1,
                                        use_bias=False, prefix='last-1x1-conv2-conv2d_'))
            self.features.add(HardSwish())

            if final_drop > 0:
                self.features.add(nn.Dropout(final_drop))
            self.output = nn.HybridSequential(prefix='output_')
            with self.output.name_scope():
                self.output.add(
                    nn.Conv2D(in_channels=cls_ch_expand, channels=classes,
                              kernel_size=1, prefix='fc_'),
                    nn.Flatten())

    def _make_layer(self, kernel_size, exp_ch, out_channel, use_se, act_func, asymmrate, stride=1,
                    prefix='', **kwargs):

        mid_planes = exp_ch
        out_planes = out_channel
        layer = AsymmBottleneck(self.inplanes, mid_planes,
                                out_planes, kernel_size, asymmrate,
                                act_func, strides=stride, use_se=use_se, prefix=prefix, **kwargs)
        self.inplanes = out_planes
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


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


def get_asymm_net(cfgs, multiplier=1., asymmrate=1, pretrained=False, ctx=cpu(), norm_layer=BatchNorm,
                  norm_kwargs=None, dropout=0.2,
                  **kwargs):
    net = AsymmNet(*cfgs, multiplier=multiplier, asymmrate=asymmrate, final_drop=dropout,
                   norm_layer=norm_layer, **kwargs)
    return net


if __name__ == '__main__':
    from utils.utils import plot_network, vis_netron

    cfgs = get_asymmnet_cfgs('asymmnet_large')
    net = get_asymm_net(cfgs, multiplier=1., classes=1000, asymmrate=0, shuffle_group=0)
    # print(net)
    # vis_netron(net, './', 'asym0')
    plot_network(net, './')
    net = get_asymm_net(cfgs, multiplier=1., classes=1000, asymmrate=1, shuffle_group=0)
    # vis_netron(net, './', 'asym1')
    plot_network(net, './')
    net = get_asymm_net(cfgs, multiplier=1., classes=1000, asymmrate=2, shuffle_group=0)
    # vis_netron(net, './', 'asym2')
    plot_network(net, './')

