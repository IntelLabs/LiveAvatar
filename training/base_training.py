from __future__ import annotations

import glob
import time

import datetime
import os
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as td
import torch.nn.modules.distance
from torchmetrics import MetricCollection
from accelerate import Accelerator

from utils.util import makedirs
import utils.log as log

eps = 1e-8


# save some samples to visualize the training progress
def get_fixed_samples(dl, num):
    assert isinstance(dl, td.DataLoader)
    dl = td.DataLoader(dl.dataset, batch_size=num, shuffle=False, num_workers=0)
    return next(iter(dl))


def __reduce(errs, reduction):
    if reduction == 'mean':
        return errs.mean()
    elif reduction == 'sum':
        return errs.sum()
    elif reduction == 'none':
        return errs
    else:
        raise ValueError("Invalid parameter reduction={}".format(reduction))


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data)
        torch.nn.init.xavier_uniform(m)
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight)


class Training(object):

    def __init__(
            self,
            dataloaders,
            net: nn.Module,
            lr: float = 1e-4,
            session_name: str = 'debug',
            snapshot_dir: str = "./results/checkpoints",
            print_freq: int = 10,
            save_freq: int = 10,
            eval_freq: int = -1,
            vis_freq: int = -1,
            resume: str | None = None,
            seed: int | None = None,
            config = None,
            accelerator: Accelerator | None = None,
            **kwargs
    ):
        self.seed = seed

        self.session_name = session_name
        self.dataloaders = dataloaders
        self.net = net
        self.device = net.device
        self.print_freq = print_freq
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.vis_freq = vis_freq
        self.lr = lr
        self.resume = resume

        self.snapshot_dir = snapshot_dir
        self.max_epochs = 999
        self.total_iter = 0
        self.total_items = 0
        self.iter_in_epoch = 0
        self.epoch = 0
        self.best_score = 999
        self.epoch_stats = []
        self.figures = {}
        self.metrics = MetricCollection([])
        self._stats_message_head_length = 10

        self.total_training_time_previous = 0
        self.time_start_training = time.time()

        self.cfg = config
        self.accelerator = accelerator

        if resume is not None:

            if resume == 'last':
                # Resume from latest snapshot.
                # Looking for files that match pattern '<snapshot_dir>/<session_name>/epoch_XXXXX.pth'
                # resume = sorted(glob.glob(os.path.join(self.snapshot_dir, self.session_name, '*.pth')))[-1]
                files = glob.glob(os.path.join(self.snapshot_dir, self.session_name, '*.pth'))
                timestamps = np.argsort([os.path.getmtime(f) for f in files])
                resume = os.path.abspath(files[timestamps[-1]])

            self.log("")
            self.log("Resuming session {} from snapshot {}...".format(self.session_name, resume))
            self._load_snapshot(resume)


    @property
    def is_main_process(self):
        if self.accelerator is None:
            return True
        return self.accelerator.is_local_main_process

    def _init_metrics(self):
        self.metrics.reset()
        self.epoch_stats = []

    def _evaluate_metrics(self, metrics: list[dict] | None):
        pass

    def _is_snapshot_iter(self):
        return self.is_main_process and self.save_freq > 0 and (self.total_iter+1) % self.save_freq == 0 and (self.total_iter+1) > 0

    def _print_interval(self, eval):
        return self.print_freq if eval else self.print_freq

    def _is_printout_iter(self, eval):
        return self.is_main_process and (self.iter_in_epoch+1) % self._print_interval(eval) == 0

    def _is_eval_epoch(self):
        return self.eval_freq > 0 and (self.epoch+1) % self.eval_freq == 0 and 'val' in self.dataloaders

    def _training_time(self):
        return int(time.time() - self.time_start_training)

    def total_training_time(self):
        return self.total_training_time_previous + self._training_time()

    def _print_iter_stats(self, stats):
        raise NotImplementedError

    def _print_epoch_summary(self, epoch_stats, epoch_starttime):
        raise NotImplementedError

    def _run_batch(self, data, is_eval, visualize: bool = False):
        raise NotImplementedError

    def _on_epoch_end(self, is_eval):
        pass

    def log(self, msg):
        if self.is_main_process:
            log.info(msg)

    def train(self):
        self.log("")
        self.log("Learning rate: {}".format(self.lr))
        self.log("Batch size per GPU: {}".format(self.cfg.batchsize))
        self.log("Number of GPUs: {}".format(self.accelerator.num_processes))
        self.log("Effective batch size: {}".format(self.cfg.batchsize * self.accelerator.num_processes))
        self.log("Workers: {}".format(self.dataloaders['train'].num_workers))

        self.net.train()

        self.log("")
        self.log("Training '{}'...".format(self.session_name))
        # self.log("")

        while self.max_epochs is None or self.epoch < self.max_epochs:
            self.log('')
            self.log('Epoch {}/{}'.format(self.epoch + 1, self.max_epochs))
            self.log('=' * 10)

            self._init_metrics()
            epoch_starttime = time.time()
            self._run_epoch(self.dataloaders['train'])

            # save model every few epochs
            if self.save_freq > 0 and (self.epoch + 1) % self.save_freq == 0:
                self._save_snapshot(is_best=False)

            # print average loss and accuracy over epoch
            self._print_epoch_summary(self.epoch_stats, epoch_starttime)
            # self._evaluate_metrics()

            if self._is_eval_epoch():
                with torch.no_grad():
                    self.validate()

            self.epoch += 1

        time_elapsed = time.time() - self.time_start_training
        self.log('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def validate(self):
        self.log("")
        self.log("Validating '{}'...".format(self.session_name))

        epoch_starttime = time.time()
        self._init_metrics()
        self.net.eval()

        self._run_epoch(self.dataloaders['val'], is_eval=True)

        # print average loss and accuracy over epoch
        self._print_epoch_summary(self.epoch_stats, epoch_starttime)

        # self._evaluate_metrics()
        return self.epoch_stats

    def _run_epoch(self, dataloader: td.DataLoader, is_eval=False):
        iter_endtime = time.time()

        batch_size = dataloader.batch_sampler.batch_size
        self.iters_per_epoch = len(dataloader)
        self.iter_in_epoch = 0

        for data in dataloader:
            time_dataloading = time.time() - iter_endtime

            # run prediction and backprop
            t_proc_start = time.time()
            is_vis_iter = self.vis_freq > 0 and self.iter_in_epoch % self.vis_freq == 0
            iter_stats, figures = self._run_batch(data, is_eval=is_eval, visualize=is_vis_iter)
            time_processing = time.time() - t_proc_start

            # statistics
            self.total_items += batch_size
            iter_stats.update({
                'epoch': self.epoch,
                'iter': self.iter_in_epoch,
                'total_iter': self.total_iter,
                'timestamp': time.time(),
                'time_dataloading': time_dataloading,
                'time_processing': time_processing,
                'iter_time': time.time() - iter_endtime,
            })
            self.epoch_stats.append(iter_stats)

            # print stats every N batches
            if self._is_printout_iter(is_eval):
                self._print_iter_stats(self.epoch_stats[-self._print_interval(is_eval):])

            if not is_eval:
                self.total_iter += 1
                self.net.total_iter = self.total_iter
            self.iter_in_epoch += 1

            iter_endtime = time.time()

        self._on_epoch_end(is_eval)

    def _save_snapshot(self, is_best=False):
        if not self.is_main_process:
            return

        def write_file(filepath, model):
            meta=dict(
                epoch=self.epoch + 1,
                total_iter=self.total_iter,
                total_time=self.total_training_time(),
                best_score=self.best_score,
                seed=self.seed,
                experiment_name=self.session_name
            )
            state_dict = self.accelerator.unwrap_model(model).state_dict()

            # remove (frozen) dino weights from state dict to reduce file size
            for k in list(state_dict.keys()):
                if k[:5] == 'dino.':
                    del state_dict[k]

            snapshot=dict(
                arch=type(model).__name__,
                state_dict=state_dict,
                meta=meta
            )
            makedirs(filepath)
            self.accelerator.save(snapshot, filepath)

        snapshot_name = os.path.join(self.session_name, 'epoch_{:05d}.pth'.format(self.epoch+1))
        output_dir =  os.path.join(self.snapshot_dir, snapshot_name)
        write_file(output_dir, self.net)
        self.log(f"*** saved checkpoint {output_dir} *** ")

        # save a copy of this snapshot as the best one so far
        # if is_best:
        #     io_utils.copy_files(src_dir=model_snap_dir, dst_dir=model_data_dir, pattern='*.mdl')

    def _load_snapshot(self, filename):

        if os.path.isabs(filename):
            filepath = filename
        else:
            filepath = os.path.join(self.snapshot_dir, filename)

        if os.path.splitext(filepath)[1] != '.pth':
            filepath = filepath + '.pth'

        snapshot = torch.load(filepath, weights_only=True, map_location="cpu")

        try:
            self.net.load_state_dict(snapshot['state_dict'], strict=False)
        except RuntimeError as e:
            print(e)

        meta = snapshot['meta']
        self.epoch = meta['epoch']
        self.total_iter = meta['total_iter']
        self.total_training_time_previous = meta.get('total_time', 0)
        self.total_items = meta.get('total_items', 0)
        self.best_score = meta['best_score']
        self.net.total_iter = self.total_iter
        str_training_time = str(datetime.timedelta(seconds=self.total_training_time()))

        self.log("Model {} trained for {} iterations ({}).".format(
            filename, self.total_iter, str_training_time)
        )
