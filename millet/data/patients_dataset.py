# -*- coding: utf-8 -*-
import pickle
import numpy as np
import pandas as pd
import torch
from overrides import override
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch.nn.functional as F
import sys, argparse, os
from typing import List, Tuple
from aeon.datasets import load_classification
from millet.data.mil_tsc_dataset import MILTSCDataset
from millet.util import load_ts_file

class PatientDataset(MILTSCDataset):
    def __init__(self, args, dataset_name: str, split: str, seed=0):
        super().__init__(args, dataset_name, split)

    def get_time_series_collection_and_targets(self, split: str) -> Tuple[List[torch.Tensor], torch.Tensor]:
        if split != '':
            path = "./data/patients/{:s}_{:s}.ts".format(self.dataset_name, split.upper())
        else:
            path = "./data/patients/{:s}.ts".format(self.dataset_name)
        samples, labels = load_ts_file(path)
        labels_list = [int(t.item()) for t in labels]
        self.label = F.one_hot(torch.tensor(labels_list)).float()
        self.labels = labels
        self.FeatList = samples

        # elif split == 'validation':
        #     self.FeatList = X_val
        #     self.label = y_val
        self.feat_in = self.FeatList[0].shape[0]
        self.max_len = self.args.max_seq
        self.num_class =  self.args.n_clz
        print(f"get samples {split}--{len(self.label)}")
        return self.FeatList, self.label

    @override
    def _get_n_clz(self) -> int:
        return self.num_class

    @override
    def __getitem__(self, idx):
        # print(torch.from_numpy(self.FeatList[idx]).shape)
        # print(torch.from_numpy(self.FeatList[idx]).squeeze(0).shape)
        feats = self.FeatList[idx] #L*d
        min_len =self.args.max_seq
        # feats = F.pad(feats, pad=(0, 0, min_len-feats.shape[0], 0))
        label = self.label[idx].float()

        return {
            "bag":feats,
            "target":label
        }

    def __len__(self):
         return len(self.label)

    def proterty(self):
        return self.max_len,self.num_class,self.feat_in
