# ------------------------------------------------------------------------
# GRIT: Faster and Better Image captioning Transformer
# Licensed under the Creative Commons Attribution.
# ------------------------------------------------------------------------
# Modified from Meshed Memory Transformer
# https://github.com/aimagelab/meshed-memory-transformer
# ------------------------------------------------------------------------
from torch import nn
from utils.typing import *
from contextlib import contextmanager


# 这段代码定义了一个自定义的 PyTorch 模块类 Module，它扩展了 PyTorch 的 nn.Module，并添加了一些额外的功能，主要用于管理模块的内部状态。


class Module(nn.Module):

    def __init__(self):
        super(Module, self).__init__()
        self._is_stateful = False  # Ture 通常表示“有状态的”。在编程中，“有状态”意味着对象会保留某些状态信息，可能会随着时间或操作而改变。false，“无状态”对象不保留状态，通常是静态的或一次性的。
        self._state_names = []
        self._state_defaults = dict()
        self.timestep = 0


    # 作用：注册一个状态（state），并将其存储为模块的缓冲区（buffer）。
    # 缓冲区（buffer）： 记录中间计算结果或其他需要持久化的值，不会在优化器中更新。
    def register_state(self, name: str, default: TensorOrNone):
        self._state_names.append(name)
        if default is None:
            self._state_defaults[name] = None
        else:
            self._state_defaults[name] = default.clone().detach()
        self.register_buffer(name, default)


    # 作用：生成器方法，用于迭代模块及其子模块的所有状态
    def states(self):
        for name in self._state_names:
            yield self._buffers[name]
        for m in self.children():
            if isinstance(m, Module):
                yield from m.states()

    def apply_to_states(self, fn):
        for name in self._state_names:
            self._buffers[name] = fn(self._buffers[name])   #在beamsearch中复制注册变量gri_feat等
        for m in self.children():
            if isinstance(m, Module):
                m.apply_to_states(fn)

    def _init_states(self, batch_size: int):
        for name in self._state_names:
            if self._state_defaults[name] is None:
                self._buffers[name] = None
            else:
                self._buffers[name] = self._state_defaults[name].clone().detach().to(self._buffers[name].device)
                self._buffers[name] = self._buffers[name].unsqueeze(0)
                self._buffers[name] = self._buffers[name].expand([batch_size,] + list(self._buffers[name].shape[1:]))
                self._buffers[name] = self._buffers[name].contiguous()

    def _reset_states(self):
        for name in self._state_names:
            if self._state_defaults[name] is None:
                self._buffers[name] = None
            else:
                self._buffers[name] = self._state_defaults[name].clone().detach().to(self._buffers[name].device)

    def enable_statefulness(self, batch_size: int):
        for m in self.children():
            if isinstance(m, Module):
                m.enable_statefulness(batch_size)
        self._init_states(batch_size)
        self._is_stateful = True

    def disable_statefulness(self):
        self.timestep = 0
        for m in self.children():
            if isinstance(m, Module):
                m.disable_statefulness()
        self._reset_states()
        self._is_stateful = False


    @contextmanager
    def statefulness(self, batch_size: int):
        self.enable_statefulness(batch_size)
        try:
            yield
        finally:
            self.disable_statefulness()


class ModuleList(nn.ModuleList, Module):
    pass


class ModuleDict(nn.ModuleDict, Module):
    pass


