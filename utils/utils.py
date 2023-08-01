import types
import math
from torch import inf
from functools import wraps
import warnings
import weakref
from collections import Counter
from bisect import bisect_right

from torch.optim.optimizer import Optimizer
import torch


class TrapezoidLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self,
                 optimizer,
                 base_lr,
                 max_lr,
                 step_size_up=2000,
                 step_size_down=None,
                 mode='triangular',
                 gamma=1.,
                 scale_fn=None,
                 scale_mode='cycle',
                 cycle_momentum=True,
                 base_momentum=0.8,
                 max_momentum=0.9,
                 last_epoch=-1,
                 hard_bound=None,
                 verbose=False):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        ###
        self.hard_bound = hard_bound if hard_bound is not None else max_lr
        ###

        base_lrs = self._format_param('base_lr', optimizer, base_lr)
        if last_epoch == -1:
            for lr, group in zip(base_lrs, optimizer.param_groups):
                group['lr'] = lr

        self.max_lrs = self._format_param('max_lr', optimizer, max_lr)

        step_size_up = float(step_size_up)
        step_size_down = float(step_size_down) if step_size_down is not None else step_size_up
        self.total_size = step_size_up + step_size_down
        self.step_ratio = step_size_up / self.total_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        self._scale_fn_ref = None
        self._scale_fn_custom = scale_fn
        self.scale_mode = scale_mode
        self._init_scale_fn()

        self.cycle_momentum = cycle_momentum
        if cycle_momentum:
            if 'momentum' not in optimizer.defaults:
                raise ValueError('optimizer must support momentum with `cycle_momentum` option enabled')

            base_momentums = self._format_param('base_momentum', optimizer, base_momentum)
            if last_epoch == -1:
                for momentum, group in zip(base_momentums, optimizer.param_groups):
                    group['momentum'] = momentum
            self.base_momentums = [group['momentum'] for group in optimizer.param_groups]
            self.max_momentums = self._format_param('max_momentum', optimizer, max_momentum)

        super().__init__(optimizer, last_epoch, verbose)
        self.base_lrs = base_lrs

    def _init_scale_fn(self):
        if self._scale_fn_custom is not None:
            return
        if self.mode == 'triangular':
            self._scale_fn_ref = weakref.WeakMethod(self._triangular_scale_fn)
            self.scale_mode = 'cycle'
        elif self.mode == 'triangular2':
            self._scale_fn_ref = weakref.WeakMethod(self._triangular2_scale_fn)
            self.scale_mode = 'cycle'
        elif self.mode == 'exp_range':
            self._scale_fn_ref = weakref.WeakMethod(self._exp_range_scale_fn)
            self.scale_mode = 'iterations'

    def _format_param(self, name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(param)))
            return param
        else:
            return [param] * len(optimizer.param_groups)

    def scale_fn(self, x):
        if self._scale_fn_custom is not None:
            return self._scale_fn_custom(x)

        else:
            return self._scale_fn_ref()(x)

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma ** (x)

    def get_lr(self):

        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        cycle = math.floor(1 + self.last_epoch / self.total_size)
        x = 1. + self.last_epoch / self.total_size - cycle
        if x <= self.step_ratio:
            scale_factor = x / self.step_ratio
        else:
            scale_factor = (x - 1) / (self.step_ratio - 1)

        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            base_height = (max_lr - base_lr) * scale_factor
            if self.scale_mode == 'cycle':
                lr = min(self.hard_bound, base_lr + base_height * self.scale_fn(cycle))
            else:
                lr = min(self.hard_bound, base_lr + base_height * self.scale_fn(self.last_epoch))
            lrs.append(lr)

        if self.cycle_momentum:
            momentums = []
            for base_momentum, max_momentum in zip(self.base_momentums, self.max_momentums):
                base_height = (max_momentum - base_momentum) * scale_factor
                if self.scale_mode == 'cycle':
                    momentum = max_momentum - base_height * self.scale_fn(cycle)
                else:
                    momentum = max_momentum - base_height * self.scale_fn(self.last_epoch)
                momentums.append(momentum)
            for param_group, momentum in zip(self.optimizer.param_groups, momentums):
                param_group['momentum'] = momentum

        return lrs

    def state_dict(self):
        state = super().state_dict()
        # We are dropping the `_scale_fn_ref` attribute because it is a `weakref.WeakMethod` and can't be pickled
        state.pop("_scale_fn_ref")
        return state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._init_scale_fn()