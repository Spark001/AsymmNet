# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable= arguments-differ,unused-argument,missing-docstring,too-many-function-args
"""MobileNet and MobileNetV2, implemented in Gluon."""

from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock

import sys, os

sys.path.insert(0, os.getcwd())


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


# pylint: disable= too-many-arguments
def _add_conv(out, channels=1, kernel=1, stride=1, pad=0,
              num_group=1, active=True, relu6=False, norm_layer=BatchNorm, norm_kwargs=None):
    out.add(nn.Conv2D(channels, kernel, stride, pad, groups=num_group, use_bias=False))
    out.add(norm_layer(scale=True, **({} if norm_kwargs is None else norm_kwargs)))
    if active:
        out.add(ReLU6() if relu6 else nn.Activation('relu'))


def _add_conv_dw(out, dw_channels, channels, stride, relu6=False,
                 norm_layer=BatchNorm, norm_kwargs=None):
    _add_conv(out, channels=dw_channels, kernel=3, stride=stride,
              pad=1, num_group=dw_channels, relu6=relu6,
              norm_layer=norm_layer, norm_kwargs=norm_kwargs)
    _add_conv(out, channels=channels, relu6=relu6,
              norm_layer=norm_layer, norm_kwargs=norm_kwargs)


class LinearBottleneck(nn.HybridBlock):
    r"""LinearBottleneck used in MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    channels : int
        Number of output channels.
    t : int
        Layer expansion ratio.
    stride : int
        stride
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """

    def __init__(self, in_channels, channels, t, stride,
                 norm_layer=BatchNorm, norm_kwargs=None, asymmrate=0, **kwargs):
        super(LinearBottleneck, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels == channels
        with self.name_scope():
            self.out = nn.HybridSequential()

            if t != 1:
                _add_conv(self.out,
                          in_channels * t,
                          relu6=True,
                          norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            _add_conv(self.out,
                      in_channels * t,
                      kernel=3,
                      stride=stride,
                      pad=1,
                      num_group=in_channels * t,
                      relu6=True,
                      norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            _add_conv(self.out,
                      channels,
                      active=False,
                      relu6=True,
                      norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x):
        out = self.out(x)
        if self.use_shortcut:
            out = F.elemwise_add(out, x)
        return out


class LinearBottleneckAsymm(nn.HybridBlock):
    r"""LinearBottleneck used in MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    channels : int
        Number of output channels.
    t : int
        Layer expansion ratio.
    stride : int
        stride
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """

    def __init__(self, in_channels, channels, t, stride,
                 norm_layer=BatchNorm, asymmrate=0, norm_kwargs=None, **kwargs):
        super(LinearBottleneckAsymm, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels == channels
        with self.name_scope():
            self.out = nn.HybridSequential()

            num_mid = in_channels * t
            self.t = t
            self.asymmrate = asymmrate

            if self.t != 1:
                self.expand = nn.HybridSequential()
                _add_conv(self.expand,
                          in_channels * t - in_channels * asymmrate,
                          relu6=True,
                          norm_layer=norm_layer, norm_kwargs=norm_kwargs)
                num_mid += in_channels * asymmrate
            _add_conv(self.out,
                      num_mid,
                      kernel=3,
                      stride=stride,
                      pad=1,
                      num_group=num_mid,
                      relu6=True,
                      norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            _add_conv(self.out,
                      channels,
                      active=False,
                      relu6=True,
                      norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x):
        if self.t != 1:
            out = self.expand(x)
            feat = []
            for _ in range(self.asymmrate):
                feat.append(x)
            feat.append(out)
            for _ in range(self.asymmrate):
                feat.append(x)
            out = F.concat(*feat, dim=1)
        else:
            out = x
        out = self.out(out)
        if self.use_shortcut:
            out = F.elemwise_add(out, x)
        return out


# Net
class MobileNet(HybridBlock):
    r"""MobileNet model from the
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861>`_ paper.

    Parameters
    ----------
    multiplier : float, default 1.0
        The width multiplier for controlling the model size. Only multipliers that are no
        less than 0.25 are supported. The actual number of channels is equal to the original
        channel size multiplied by this multiplier.
    classes : int, default 1000
        Number of classes for the output layer.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """

    def __init__(self, multiplier=1.0, classes=1000,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(MobileNet, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            with self.features.name_scope():
                _add_conv(self.features, channels=int(32 * multiplier), kernel=3, pad=1, stride=2,
                          norm_layer=norm_layer, norm_kwargs=norm_kwargs)
                dw_channels = [int(x * multiplier) for x in [32, 64] + [128] * 2 +
                               [256] *
                               2 +
                               [512] *
                               6 +
                               [1024]]
                channels = [int(x * multiplier) for x in [64] +
                            [128] * 2 + [256] * 2 + [512] * 6 + [1024] * 2]
                strides = [1, 2] * 3 + [1] * 5 + [2, 1]
                for dwc, c, s in zip(dw_channels, channels, strides):
                    _add_conv_dw(self.features, dw_channels=dwc, channels=c, stride=s,
                                 norm_layer=norm_layer, norm_kwargs=norm_kwargs)
                self.features.add(nn.GlobalAvgPool2D())
                self.features.add(nn.Flatten())

            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


class MobileNetV2(nn.HybridBlock):
    r"""MobileNetV2 model from the
    `"Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381>`_ paper.
    Parameters
    ----------
    multiplier : float, default 1.0
        The width multiplier for controlling the model size. The actual number of channels
        is equal to the original channel size multiplied by this multiplier.
    classes : int, default 1000
        Number of classes for the output layer.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """

    def __init__(self, multiplier=1.0, classes=1000,
                 norm_layer=BatchNorm, norm_kwargs=None, asymmrate=0, bottleneck=LinearBottleneck, **kwargs):
        super(MobileNetV2, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='features_')
            with self.features.name_scope():
                _add_conv(self.features, int(32 * multiplier), kernel=3,
                          stride=2, pad=1, relu6=True,
                          norm_layer=norm_layer, norm_kwargs=norm_kwargs)

                in_channels_group = [int(x * multiplier) for x in [32] + [16] + [24] * 2
                                     + [32] * 3 + [64] * 4 + [96] * 3 + [160] * 3]
                channels_group = [int(x * multiplier) for x in [16] + [24] * 2 + [32] * 3
                                  + [64] * 4 + [96] * 3 + [160] * 3 + [320]]
                ts = [1] + [6] * 16
                strides = [1, 2] * 2 + [1, 1, 2] + [1] * 6 + [2] + [1] * 3

                for in_c, c, t, s in zip(in_channels_group, channels_group, ts, strides):
                    self.features.add(bottleneck(in_channels=in_c,
                                                 channels=c,
                                                 t=t,
                                                 stride=s,
                                                 norm_layer=norm_layer,
                                                 asymmrate=asymmrate,
                                                 norm_kwargs=norm_kwargs))

                last_channels = int(1280 * multiplier) if multiplier > 1.0 else 1280
                _add_conv(self.features,
                          last_channels,
                          relu6=True,
                          norm_layer=norm_layer, norm_kwargs=norm_kwargs)

                self.features.add(nn.GlobalAvgPool2D())

            self.output = nn.HybridSequential(prefix='output_')
            with self.output.name_scope():
                self.output.add(
                    nn.Conv2D(classes, 1, use_bias=False, prefix='pred_'),
                    nn.Flatten())

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_mobilenetv2(multiplier=1.0):
    return MobileNetV2(multiplier=multiplier, bottleneck=LinearBottleneck)


def get_mobilenetv2_asymm(multiplier=1.0, asymmrate=0):
    return MobileNetV2(multiplier=multiplier, bottleneck=LinearBottleneckAsymm, asymmrate=asymmrate)


if __name__ == '__main__':
    from utils.utils import plot_network

    net = get_mobilenetv2_asymm(multiplier=1.0, asymmrate=1)
    # net = get_mobilenetv2()
    print(net)
    plot_network(net, './')
