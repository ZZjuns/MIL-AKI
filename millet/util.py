"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
import logging
import os
import sys
from functools import partial

from torch import nn
from tqdm import tqdm
import pickle
""" Lookahead Optimizer Wrapper.
Implementation modified from: https://github.com/alphadl/lookahead.pytorch
Paper: `Lookahead Optimizer: k steps forward, 1 step back` - https://arxiv.org/abs/1907.08610
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from torch.optim.optimizer import Optimizer
from collections import defaultdict



# tqdm wrapper to write to stdout rather than stderr
custom_tqdm = partial(tqdm, file=sys.stdout)

def get_gpu_device_for_os() -> torch.device:
    """
    Get GPU device for different operating systems. Currently setup for MacOS and Linux.

    :return: Torch GPU device.
    """
    if sys.platform == "darwin":
        return torch.device("mps")
    elif sys.platform == "linux":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("Cuda GPU device not found on Linux.")
    raise NotImplementedError("GPU support not configured for platform {:s}".format(sys.platform))


def cross_entropy_criterion(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Cross entropy criterion wrapper that ensures targets are integer class indices.

    :param predictions: Tensor of logits.
    :param targets: Tensor of targets.
    :return:
    """
    loss: float = nn.CrossEntropyLoss()(predictions, targets.long())
    return loss

def generate_variable_windows(data, max_window_size=48, pred_horizon=6):
    """
    基于变长窗口和滑动窗口构建样本：
    - 前48小时：逐渐增加窗口长度（从1增加到48）
    - 48小时后：使用固定48小时窗口滑动构建样本

    参数：
    - data：输入的时间序列DataFrame（按stay_id和hour_time排序）
    - max_window_size：窗口最大长度（默认48小时）
    - pred_horizon：预测未来6小时的窗口（默认6小时）

    返回：
    - samples：所有窗口样本的数据列表（每个样本为DataFrame）
    - labels：样本对应的标签（0或1）
    """
    samples = []
    labels = []

    # 按stay_id分组，处理每个病人的时间序列
    for stay_id, group in data.groupby('stay_id'):
        group = group.reset_index(drop=True)  # 重置索引，方便滑动窗口处理
        # 遍历每个可能的窗口起始位置
        for i in range(len(group) - pred_horizon):
            # 动态调整窗口大小：最小为1，最大为48小时
            window_size = min(i + 1, max_window_size)

            # 提取当前的窗口数据
            window = group.iloc[i - window_size + 1:i + 1]

            # 检查未来6小时内是否有AKI
            future_aki = group.iloc[i + pred_horizon]['aki_stage']
            # label = 1 if future_aki > 0 else 0  # 若未来6小时有AKI，则标记为1
            label = future_aki

            # 保存样本（去掉无关列）
            # samples.append(window.drop(columns=['subject_id', 'stay_id', 'intime', 'outtime', 'first_aki_time', 'aki_stage']))
            sample_array = window.drop(columns=['subject_id', 'stay_id', 'intime', 'outtime', 'first_aki_time', 'hour_time', 'aki_stage']).to_numpy()
            samples.append(sample_array)
            labels.append(label)
    # 转换为 NumPy 数组
    return samples, labels

def load_ts_file(file_path):
    """
    从 .ts 文件中读取数据，并返回 samples 和 labels 列表。
    """
    samples = []
    labels = []

    with open(file_path, 'rb') as f:
        while True:
            try:
                # 逐条反序列化读取数据
                record = pickle.load(f)
                sample_tensor = torch.tensor(record['data']).float()
                label_tensor = torch.tensor(record['label']).float()
                samples.append(sample_tensor)
                labels.append(label_tensor)  # 将标签添加到 labels 列表
            except EOFError:
                # 读取到文件末尾，退出循环
                break
    return samples, labels


class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.defaults.update(defaults)
        self.state = defaultdict(dict)
        # manually add our defaults to the param groups
        for name, default in defaults.items():
            for group in self.param_groups:
                group.setdefault(name, default)
        # 添加缺失的属性，确保与较新版本的 PyTorch 兼容
        self._optimizer_step_post_hooks = getattr(base_optimizer, '_optimizer_step_post_hooks', {})
        self._optimizer_step_pre_hooks = getattr(base_optimizer, '_optimizer_step_pre_hooks', {})
    def update_slow(self, group):
        for fast_p in group["params"]:
            if fast_p.grad is None:
                continue
            param_state = self.state[fast_p]
            if 'slow_buffer' not in param_state:
                param_state['slow_buffer'] = torch.empty_like(fast_p.data)
                param_state['slow_buffer'].copy_(fast_p.data)
            slow = param_state['slow_buffer']
            # slow.add_(group['lookahead_alpha'], fast_p.data - slow)
            slow.add_(fast_p.data - slow, alpha=group['lookahead_alpha'])
            fast_p.data.copy_(slow)

    def sync_lookahead(self):
        for group in self.param_groups:
            self.update_slow(group)

    def step(self, closure=None):
        #assert id(self.param_groups) == id(self.base_optimizer.param_groups)
        loss = self.base_optimizer.step(closure)
        for group in self.param_groups:
            group['lookahead_step'] += 1
            if group['lookahead_step'] % group['lookahead_k'] == 0:
                self.update_slow(group)
        return loss

    def state_dict(self):
        fast_state_dict = self.base_optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict['state']
        param_groups = fast_state_dict['param_groups']
        return {
            'state': fast_state,
            'slow_state': slow_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        fast_state_dict = {
            'state': state_dict['state'],
            'param_groups': state_dict['param_groups'],
        }
        self.base_optimizer.load_state_dict(fast_state_dict)

        # We want to restore the slow state, but share param_groups reference
        # with base_optimizer. This is a bit redundant but least code
        slow_state_new = False
        if 'slow_state' not in state_dict:
            print('Loading state_dict from optimizer without Lookahead applied.')
            state_dict['slow_state'] = defaultdict(dict)
            slow_state_new = True
        slow_state_dict = {
            'state': state_dict['slow_state'],
            'param_groups': state_dict['param_groups'],  # this is pointless but saves code
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.param_groups = self.base_optimizer.param_groups  # make both ref same container
        if slow_state_new:
            # reapply defaults to catch missing lookahead specific ones
            for name, default in self.defaults.items():
                for group in self.param_groups:
                    group.setdefault(name, default)

def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


def make_dirs(save_dir):
    existing_versions = os.listdir(save_dir)

    if len(existing_versions) > 0:
        max_version = int(existing_versions[0].split("_")[-1])
        for v in existing_versions:
            if v.endswith('.csv'):
                continue
            ver = int(v.split("_")[-1])
            if ver > max_version:
                max_version = ver
        version = int(max_version) + 1
    else:
        version = 0

    return f"{save_dir}/exp_{version}"


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def kfold_split(dataset, k_splits):
    """
    Split the dataset into k folds
    Args:
        dataset: the list of sample features
        k_splits: the number of folds
    """
    assert len(dataset) > 0, 'Dataset is empty!'
    cv_dataset_list = []  # [(trainset_1, testset_1), ..., (trainset_k, testset_k)]

    # chunk the dataset into k folds
    dataset_size = len(dataset)
    fold_size = dataset_size / float(k_splits)
    chunked_dataset = []
    last = 0.0
    split_counter = 1
    while split_counter <= k_splits:
        chunked_dataset.append(dataset[int(last):int(last + fold_size)])
        last += fold_size
        split_counter += 1
    assert len(chunked_dataset) == k_splits, 'The size of chunked_dataset should be same as k_splits!'

    for index in range(k_splits):
        testset = chunked_dataset[index]
        trainset = []
        for i in range(k_splits):
            if i == index:
                continue
            trainset += chunked_dataset[i]

        train_test = (trainset, testset)
        cv_dataset_list.append(train_test)
    return cv_dataset_list


join = os.path.join
