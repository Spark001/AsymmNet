import os
import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon.block import HybridBlock
import math
import numpy as np
from mxnet.contrib import onnx as onnx_mxnet
import pdb


# Two functions for reading data from record file or raw images
def get_data_rec(opt, rec_train, rec_train_idx, rec_val, rec_val_idx, batch_size, num_workers):
    rec_train = os.path.expanduser(rec_train)
    rec_train_idx = os.path.expanduser(rec_train_idx)
    rec_val = os.path.expanduser(rec_val)
    rec_val_idx = os.path.expanduser(rec_val_idx)
    jitter_param = 0.4
    lighting_param = 0.1
    input_size = opt.input_size
    crop_ratio = opt.crop_ratio if opt.crop_ratio > 0 else 0.875
    resize = int(math.ceil(input_size / crop_ratio))
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]

    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        return data, label

    train_data = mx.io.ImageRecordIter(
        path_imgrec=rec_train,
        path_imgidx=rec_train_idx,
        preprocess_threads=num_workers,
        shuffle=True,
        batch_size=batch_size,

        data_shape=(3, input_size, input_size),
        mean_r=mean_rgb[0],
        mean_g=mean_rgb[1],
        mean_b=mean_rgb[2],
        std_r=std_rgb[0],
        std_g=std_rgb[1],
        std_b=std_rgb[2],
        rand_mirror=True,
        random_resized_crop=True,
        max_aspect_ratio=4. / 3.,
        min_aspect_ratio=3. / 4.,
        max_random_area=1,
        min_random_area=0.08,
        brightness=jitter_param,
        saturation=jitter_param,
        contrast=jitter_param,
        pca_noise=lighting_param,
    )
    val_data = mx.io.ImageRecordIter(
        path_imgrec=rec_val,
        path_imgidx=rec_val_idx,
        preprocess_threads=num_workers,
        shuffle=False,
        batch_size=batch_size,

        resize=resize,
        data_shape=(3, input_size, input_size),
        mean_r=mean_rgb[0],
        mean_g=mean_rgb[1],
        mean_b=mean_rgb[2],
        std_r=std_rgb[0],
        std_g=std_rgb[1],
        std_b=std_rgb[2],
    )
    return train_data, val_data, batch_fn


def plot_network(net, out_dir, tag='', shape=(1, 3, 224, 224)):
    from visualization import print_summary_ops

    x = mx.sym.var('data')

    sym = net(x)

#     from contextlib import redirect_stdout
#     with open('{}/{}macc.txt'.format(out_dir, tag), 'w') as f:
#         with redirect_stdout(f):
#             params, macs, acts = print_summary_ops(sym, shape={"data": shape})
    params, macs, acts = print_summary_ops(sym, shape={"data": shape})
    print('params: {}M, macs: {}M, acts: {}M'.format(params, macs, acts))


def export_onnx_model(json_path, param_path, output_onnx_path, input_shape=(1, 3, 224, 224)):
    """
    Export mxnet model to onnx for visualizaion with Netron
    Note: need onnx == 1.3.0
    :param save_path:
    :param net:
    :param epoch:
    :return:
    """
    onnx_mxnet.export_model(json_path, param_path, [input_shape], np.float32, output_onnx_path)


def vis_netron(net, out_dir, model_name, shape=(1, 3, 224, 224)):
    # vis network using netron
    try:
        import pdb
        from gluoncv.utils.export_helper import export_block
        input_shape = shape[1:]  # (3, 224, 224)
        onnx_input_shape = shape  # (1, 3, 224, 224)
        net.initialize(mx.init.Uniform(), ctx=mx.cpu(0))
        export_block('%s/init' % out_dir, net, preprocess=False, data_shape=input_shape, layout='CHW', ctx=mx.cpu())
        param_path = '%s/init-0000.params' % (out_dir)
        json_path = '%s/init-symbol.json' % (out_dir)
        output_onnx_path = '%s/%s.onnx' % (out_dir, model_name)
        # pdb.set_trace()
        export_onnx_model(json_path, param_path, output_onnx_path, input_shape=onnx_input_shape)
        return
    except Exception as e:
        print(e)
        import onnx
        print('onnx version: ', onnx.__version__)
        print("May be you need onnx==1.3.0, if you encounter some problems "
              "about BN when exporting model")
