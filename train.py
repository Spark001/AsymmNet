import time, logging, os

from datetime import datetime
import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag

from mxboard import SummaryWriter
from gluoncv.utils import makedirs, LRSequential, LRScheduler

from models.asymmnet import get_asymm_net, get_asymmnet_cfgs
from models.mobilenetv3 import get_mobilenet_v3

from utils.utils import plot_network, get_data_rec
from utils.args_helper import parse_args


def list_to_device(src, context):
    if isinstance(context, mx.Context):
        context = [context]
    return [nd.array(src, ctx=c) for c in context]


def main():
    parser = parse_args()
    opt = parser.parse_args()

    save_dir = os.path.join(opt.save_dir, opt.dataset,
                            opt.model + '_' + opt.tag_name + '_' + datetime.now().strftime('%m%d-%H%M%S'))
    makedirs(save_dir)

    filehandler = logging.FileHandler(os.path.join(save_dir, opt.logging_file))
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    logger.info(opt)

    batch_size = opt.batch_size
    classes = 1000
    num_training_samples = 1281167

    if opt.dataset == 'imagenet200':
        classes = 200
        num_training_samples = 258758

    num_gpus = opt.num_gpus
    batch_size *= max(1, num_gpus)
    context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    num_workers = opt.num_workers

    lr_decay = opt.lr_decay
    lr_decay_period = opt.lr_decay_period
    if opt.lr_decay_period > 0:
        lr_decay_epoch = list(range(lr_decay_period, opt.num_epochs, lr_decay_period))
    else:
        lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')]
    lr_decay_epoch = [e - opt.warmup_epochs for e in lr_decay_epoch]
    num_batches = num_training_samples // batch_size

    lr_scheduler = LRSequential([
        LRScheduler('linear', base_lr=0, target_lr=opt.lr,
                    nepochs=opt.warmup_epochs, iters_per_epoch=num_batches),
        LRScheduler(opt.lr_mode, base_lr=opt.lr, target_lr=0,
                    nepochs=opt.num_epochs - opt.warmup_epochs,
                    iters_per_epoch=num_batches,
                    step_epoch=lr_decay_epoch,
                    step_factor=lr_decay, power=2)
    ])

    model_name = opt.model

    kwargs = {'ctx': context, 'pretrained': opt.use_pretrained, 'classes': classes}
    if opt.last_gamma:
        kwargs['last_gamma'] = True

    optimizer = 'nag'
    optimizer_params = {'wd': opt.wd, 'momentum': opt.momentum, 'lr_scheduler': lr_scheduler}
    if opt.dtype != 'float32':
        optimizer_params['multi_precision'] = True

    if 'asymmnet' in model_name:
        cfgs = get_asymmnet_cfgs(model_name)
        net = get_asymm_net(cfgs=cfgs, classes=classes, multiplier=opt.width_scale,
                            dropout=opt.dropout)
    elif 'mobilenetv3' in model_name:
        net = get_mobilenet_v3(model_name=model_name, classes=classes, multiplier=opt.width_scale,
                               dropout=opt.dropout)
    else:

        raise NotImplementedError

    if opt.mode == 'hybrid':
        logger.info(net)
        net.hybridize(static_alloc=True, static_shape=True)
        plot_network(net, save_dir)

    net.cast(opt.dtype)
    if opt.resume_params != '':
        net.load_parameters(opt.resume_params, ctx=context)

    train_data, val_data, batch_fn = get_data_rec(opt, opt.rec_train, opt.rec_train_idx,
                                                  opt.rec_val, opt.rec_val_idx,
                                                  batch_size, num_workers)

    if opt.mixup:
        train_metric = mx.metric.RMSE()
    else:
        train_metric = mx.metric.Accuracy()
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)

    save_frequency = opt.save_frequency

    def mixup_transform(label, classes, lam=1, eta=0.0):
        if isinstance(label, nd.NDArray):
            label = [label]
        res = []
        for l in label:
            y1 = l.one_hot(classes, on_value=1 - eta + eta / classes, off_value=eta / classes)
            y2 = l[::-1].one_hot(classes, on_value=1 - eta + eta / classes, off_value=eta / classes)
            res.append(lam * y1 + (1 - lam) * y2)
        return res

    def smooth(label, classes, eta=0.1):
        if isinstance(label, nd.NDArray):
            label = [label]
        smoothed = []
        for l in label:
            res = l.one_hot(classes, on_value=1 - eta + eta / classes, off_value=eta / classes)
            smoothed.append(res)
        return smoothed

    def test(ctx, val_data):
        val_data.reset()
        acc_top1.reset()
        acc_top5.reset()

        for i, batch in enumerate(val_data):
            data, label = batch_fn(batch, ctx)
            outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
            acc_top1.update(label, outputs)
            acc_top5.update(label, outputs)

        _, top1 = acc_top1.get()
        _, top5 = acc_top5.get()
        return (top1, top5)

    def train(ctx):
        if isinstance(ctx, mx.Context):
            ctx = [ctx]
        if opt.resume_params == '':
            net.initialize(mx.init.MSRAPrelu(), ctx=ctx)

        if opt.dry_run:
            dummy_input = mx.nd.random.normal(0, 1, shape=(1, 3, 224, 224))
            tic = time.time()
            for _ in range(opt.dry_run):
                net(dummy_input)
            print('speed: {:.6f} ms'.format((time.time() - tic) * 1000. / opt.dry_run))

        if opt.no_wd:
            for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
                v.wd_mult = 0.0

        trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
        if opt.resume_states != '':
            trainer.load_states(opt.resume_states)

        if opt.label_smoothing or opt.mixup:
            sparse_label_loss = False
        else:
            sparse_label_loss = True
        L = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=sparse_label_loss)

        best_val_score = 0.0

        sw = SummaryWriter(logdir=save_dir, flush_secs=2)
        for epoch in range(opt.resume_epoch, opt.num_epochs):
            tic = time.time()
            train_data.reset()
            train_metric.reset()
            btic = time.time()

            train_loss = 0
            for i, batch in enumerate(train_data):
                data, label = batch_fn(batch, ctx)

                if opt.mixup:
                    lam = np.random.beta(opt.mixup_alpha, opt.mixup_alpha)
                    if epoch >= opt.num_epochs - opt.mixup_off_epoch:
                        lam = 1
                    data = [lam * X + (1 - lam) * X[::-1] for X in data]

                    if opt.label_smoothing:
                        eta = 0.1
                    else:
                        eta = 0.0
                    label = mixup_transform(label, classes, lam, eta)

                elif opt.label_smoothing:
                    hard_label = label
                    label = smooth(label, classes)

                with ag.record():
                    outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]

                    loss = [L(yhat, y.astype(opt.dtype, copy=False)) for yhat, y in zip(outputs, label)]

                for l in loss:
                    l.backward()

                trainer.step(batch_size)

                if opt.mixup:
                    output_softmax = [nd.SoftmaxActivation(out.astype('float32', copy=False)) \
                                      for out in outputs]
                    train_metric.update(label, output_softmax)
                else:
                    if opt.label_smoothing:
                        train_metric.update(hard_label, outputs)
                    else:
                        train_metric.update(label, outputs)

                if opt.log_interval and not (i + 1) % opt.log_interval:
                    train_metric_name, train_metric_score = train_metric.get()
                    logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f\tlr=%f' % (
                        epoch, i, batch_size * opt.log_interval / (time.time() - btic),
                        train_metric_name, train_metric_score, trainer.learning_rate))
                    btic = time.time()

                train_loss += sum([l.sum().asscalar() for l in loss])

            train_metric_name, train_metric_score = train_metric.get()
            throughput = int(batch_size * i / (time.time() - tic))
            sw.add_scalar(tag='learning_rate', value=trainer.learning_rate, global_step=epoch)
            sw.add_scalar(tag='train_acc', value=train_metric_score, global_step=epoch)

            train_loss = train_loss / num_training_samples
            sw.add_scalar(tag='train_loss', value=train_loss, global_step=epoch)

            top1_val, top5_val = test(ctx, val_data)
            sw.add_scalar(tag='validation_acc', value=top1_val, global_step=epoch)
            sw.add_scalar(tag='validation_acc_top5', value=top5_val, global_step=epoch)

            logger.info('[Epoch %d] training: %s=%f' % (epoch, train_metric_name, train_metric_score))
            logger.info('[Epoch %d] speed: %d samples/sec\ttime cost: %f' % (epoch, throughput, time.time() - tic))
            logger.info('[Epoch %d] validation: top1=%f top5=%f' % (epoch, top1_val, top5_val))

            if top1_val > best_val_score:
                best_val_score = top1_val
                net.save_parameters(
                    '%s/%.4f-%s-%s-%d-best.params' % (save_dir, best_val_score, opt.dataset, model_name, epoch))
                trainer.save_states(
                    '%s/%.4f-%s-%s-%d-best.states' % (save_dir, best_val_score, opt.dataset, model_name, epoch))

            if save_frequency and save_dir and (epoch + 1) % save_frequency == 0:
                net.save_parameters('%s/%s-%s-%d.params' % (save_dir, opt.dataset, model_name, epoch))
                trainer.save_states('%s/%s-%s-%d.states' % (save_dir, opt.dataset, model_name, epoch))

        if save_frequency and save_dir:
            net.save_parameters('%s/%s-%s-%d.params' % (save_dir, opt.dataset, model_name, opt.num_epochs - 1))
            trainer.save_states('%s/%s-%s-%d.states' % (save_dir, opt.dataset, model_name, opt.num_epochs - 1))
        sw.close()

    train(context)


if __name__ == '__main__':
    main()
