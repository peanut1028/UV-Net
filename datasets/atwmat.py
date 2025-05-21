#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   atwmat.py
@Time    :   2025/01/14 11:43:41
@Author  :   LGJ 
@Version :   1.0
@Contact :   lgjhsjt@163.com
@License :   (C)Copyright 2022-2025
@Desc    :   Dataset for raw material weight prediction
'''

# here put the import lib

import pathlib
import string
import os.path as osp
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from datasets.base import BaseDataset



class ATWMATDataset(BaseDataset):

    def __init__(
        self,
        root_dir,
        mode="train",
        center_and_scale=True,
        random_rotate=False,
    ):
        """
        Load the ATWMAT dataset

        ATWMAT
        |
        ├── train.txt
            ├── file1.bin  0物料名称embeding 1板厚 
                            2孔 3折弯 4切割 5净表面积cm2 6净体积cm3
                            毛体积
            ├── ...
        ├── test.txt
        ├── bin/
        ├── ...

        Args:
            root_dir (str): Root path to the dataset
            mode (str, optional): Split (train, val, or test) to load. Defaults to "train".
            center_and_scale (bool, optional): Whether to center and scale the solid. Defaults to True.
            random_rotate (bool, optional): Whether to apply random rotations to the solid in 90 degree increments. Defaults to False.
        """
        assert mode in ("train", "val", "test")
        self.mode = mode
        self.root_dir = root_dir
        self.data_txt = osp.join(self.root_dir, self.mode + ".txt")

        self.random_rotate = random_rotate

        self.file_paths = []
        self.vars = []
        self.labels = []

        self.load_data_from_txt()

        print(f"Loading {mode} data...")
        self.load_graphs(center_and_scale)
        print("Done loading {} files".format(len(self.data)))

    def load_data_from_txt(self):
        path = pathlib.Path(self.root_dir)
        with open(self.data_txt, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line == '':
                    continue
                filename, annostr = line.rsplit('  ', 1)
                values = [float(x) for x in annostr.split(' ')]
                # assert len(values) == 11, "{} has wrong number of values".format(filename)
                filename += '.bin'
                self.file_paths.append(path / 'bin' / filename)
                self.vars.append(values[:-1])  
                self.labels.append(values[-1])

    def load_graphs(self, center_and_scale=True):
        self.data = []
        for idx in tqdm(range(len(self.file_paths))):
            sample = self.load_one_graph(idx)
            if sample is None:
                continue
            if sample["graph"].edata["x"].size(0) == 0:
                # Catch the case of graphs with no edges
                continue
            self.data.append(sample)
        # 中心化和缩放
        if center_and_scale:
            self.center_and_scale()
        self.convert_to_float32()

    def load_one_graph(self, index):
        filename = self.file_paths[index]
        if not filename.exists():
            return None
        # Load the graph using base class method
        sample = super().load_one_graph(filename)
        # Additionally get the label from the filename and store it in the sample dict
        sample["vars"] = torch.tensor(self.vars[index], dtype=torch.float32)
        sample["label"] = torch.tensor(self.labels[index], dtype=torch.float32)
        return sample

    def _collate(self, batch):
        collated = super()._collate(batch)
        collated["label"] =  torch.Tensor([x["label"] for x in batch])
        collated["vars"] =  torch.cat([x["vars"].unsqueeze(0) for x in batch], dim=0)
        return collated


