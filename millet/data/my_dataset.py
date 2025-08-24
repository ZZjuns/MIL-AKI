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


class loadorean(MILTSCDataset):
    def __init__(self,args, dataset_name: str, split: str, seed=0):
        super().__init__(args,dataset_name, split)

    def get_time_series_collection_and_targets(self, split: str) -> Tuple[List[torch.Tensor], torch.Tensor]:
 # 12*29
        if self.dataset_name == 'JapaneseVowels':
            self.seq_len = 29
        elif self.dataset_name == 'SpokenArabicDigits':
            self.seq_len = 93
        elif self.dataset_name == 'CharacterTrajectories':
            self.seq_len = 182
        elif self.dataset_name == 'InsectWingbeat':
            self.seq_len = 78
        if split in ['train']:


            if self.dataset_name == 'InsectWingbeat':
                Xtr, ytr, meta =load_classification(name='InsectWingbeat', split='train',extract_path='./data/', return_metadata=True)
            else:
                Xtr, ytr, meta = load_classification(name=self.dataset_name,split='train',  return_metadata=True)
            # print(Xtr.shape)
            word_to_idx = {}
            for i in range(len(meta['class_values'])):
                word_to_idx[meta['class_values'][i]]=i


            ytr = [word_to_idx[i] for i in ytr]
            self.label =  F.one_hot(torch.tensor(ytr)).float()
            # self.label =  torch.tensor(ytr).float()
            self.FeatList = Xtr


        elif split == 'test':
            if self.dataset_name == 'InsectWingbeat':
                Xte, yte, meta =load_classification(name='InsectWingbeat', split='test',extract_path='./data/',  return_metadata=True)
            else:
                Xte, yte, meta = load_classification(name=self.dataset_name,split='test', return_metadata=True)
            word_to_idx = {}
            for i in range(len(meta['class_values'])):
                word_to_idx[meta['class_values'][i]]=i

            # Xte =torch.from_numpy(Xte).permute(0,2,1).float()
            yte = [word_to_idx[i] for i in yte]
            self.label =  F.one_hot(torch.tensor(yte)).float()
            # self.label = F.one_hot(torch.tensor(yte)).float()
            self.FeatList = Xte
            # self.label = torch.tensor(yte).float()

        self.feat_in = self.FeatList[0].shape[0]
        self.max_len = self.seq_len
        self.num_class =  meta['class_values']

        return self.FeatList, self.label

    @override
    def _get_n_clz(self) -> int:
        return self.num_class

    @override
    def __getitem__(self, idx):
        # print(torch.from_numpy(self.FeatList[idx]).shape)
        # print(torch.from_numpy(self.FeatList[idx]).squeeze(0).shape)
        feats = torch.from_numpy(self.FeatList[idx]).permute(1,0).float() #L*d

        min_len =self.seq_len

        feats = F.pad(feats, pad=(0, 0, min_len-feats.shape[0], 0))


        label = self.label[idx].float()

        return {
            "bag":feats,
            "target":label
        }

    def __len__(self):
         return len(self.label)

    def proterty(self):
        return self.max_len,self.num_class,self.feat_in
