#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   atwmat.py
@Time    :   2025/01/14 11:43:04
@Author  :   LGJ 
@Version :   1.0
@Contact :   lgjhsjt@163.com
@License :   (C)Copyright 2022-2025
@Desc    :   dataset for machining price prediction
'''

# here put the import lib
import pathlib
import os.path as osp
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np
import joblib
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch



class ATWMATDataset(Dataset):

    def __init__(
        self,
        root_dir,
        mode="train",
        center_and_scale=False,
    ):
        """
        Load the ATWCAD dataset

        ATWCAD
        |
        ├── train.txt
            ├── file1.bin  0物料名称embeding 1主材重量 2主材单价 3主材系数 4主材费用 
                            5表面处理重 6表面处理面积 
                            7板厚 8折弯 9切割 10孔 
                            11(>10个)加工价(gt)
            ├── ...
        ├── test.txt
        ├── step/
        ├── graph/
        ├── label/
        ├── ...

        Args:
            root_dir (str): Root path to the dataset
            mode (str, optional): Split (train, val, or test) to load. Defaults to "train".
            center_and_scale (bool, optional): Whether to center and scale the solid. Defaults to True.
        """
        assert mode in ("train", "val", "test")
        self.mode = mode
        self.root_dir = root_dir
        self.data_txt = osp.join(self.root_dir, self.mode + ".txt")

        self.file_paths = []
        self.vars = []
        self.labels = []

        self.load_data_from_txt()

        print(f"Loading {mode} data...")
        self.load_graphs()
        print("Done loading {} files".format(len(self.data)))

        # Normaliza
        if center_and_scale:
            self.scale()

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
                filename += '.pt'
                self.file_paths.append(path / 'pt' / filename)
                self.vars.append(values[:1]+values[2:-1])  
                self.labels.append(values[-1])

    def load_graphs(self):
        self.data = []
        for idx in tqdm(range(len(self.file_paths))):
            sample = self.load_one_graph(idx)
            if sample is None:
                continue
            if sample["graph"].edge_attr.size(0) == 0:
                # Catch the case of graphs with no edges
                continue
            self.data.append(sample)

    def load_one_graph(self, index):
        filename = self.file_paths[index]
        if not filename.exists():
            return None
        sample = {}
        sample["graph"] = torch.load(filename)
        sample["filename"] = filename.stem
        # Additionally get the label from the filename and store it in the sample dict
        sample["vars"] = torch.tensor([self.vars[index]], dtype=torch.float32)
        sample["label"] = torch.tensor(self.labels[index], dtype=torch.float32)
        return sample

    def scale(self, filter_columns_face=[0, 2, 3, 4], filter_columns_edge=[0, 2, 5], filter_columns_var=[0, 1, 3]):
        scalerPath = osp.join(self.root_dir, "scaler.joblib")
        if osp.exists(scalerPath):
            faceScaler, edgeScaler, varScaler, labelScaler = joblib.load(scalerPath)
        else:
            faceScaler, edgeScaler = self.get_graph_scaler(filter_columns_face, filter_columns_edge)
            varScaler = self.get_vars_scaler(filter_columns_var)
            labelScaler = StandardScaler().fit(np.array(self.labels).reshape(-1, 1))
            joblib.dump((faceScaler, edgeScaler, varScaler, labelScaler), scalerPath)
        for i in range(len(self.data)):
            faceFeat = self.data[i]["graph"].x.numpy().reshape(-1, 18)
            edgeFeat = self.data[i]["graph"].edge_attr.numpy().reshape(-1, 18)
            faceFeatNoScale = faceFeat[:, filter_columns_face]
            edgeFeatNoScale = edgeFeat[:, filter_columns_edge]
            faceFeatScale = np.delete(faceFeat, filter_columns_face, axis=1)
            edgeFeatScale = np.delete(edgeFeat, filter_columns_edge, axis=1)
            faceFeatScale = faceScaler.transform(faceFeatScale)
            edgeFeatScale = edgeScaler.transform(edgeFeatScale)
            faceFeat = np.insert(faceFeatScale, filter_columns_face, faceFeatNoScale, axis=1)
            edgeFeat = np.insert(edgeFeatScale, filter_columns_edge, edgeFeatNoScale, axis=1)
            self.data[i]["graph"].x = torch.from_numpy(faceFeat)
            self.data[i]["graph"].edge_attr = torch.from_numpy(edgeFeat)

            vars = self.data[i]["vars"].numpy().reshape(-1, 7)
            varsNoScale = vars[:, filter_columns_var]
            varsScale = np.delete(vars, filter_columns_var, axis=1)
            varsScale = varScaler.transform(varsScale)
            mask = np.zeros(vars.shape[1], dtype=bool)
            mask[filter_columns_var] = True
            vars[:, ~mask] = varsScale
            vars[:, mask] = varsNoScale
            self.data[i]["vars"] = torch.from_numpy(vars)

            label = self.data[i]["label"].numpy()
            label = labelScaler.transform([[label]])[0][0]
            self.data[i]["label"] = torch.tensor(label, dtype=torch.float32)

    def get_graph_scaler(self, filter_columns_face=None, filter_columns_edge=None):
        # collate graph attributes
        faceAttrs = []
        edgeAttrs = []
        for graph in self.data:
            faceAttrs.append(graph["graph"].x.numpy().reshape(-1, 18))
            edgeAttrs.append(graph["graph"].edge_attr.numpy().reshape(-1, 18))
        faceAttrs = np.concatenate(faceAttrs, axis=0)
        edgeAttrs = np.concatenate(edgeAttrs, axis=0)
        # filter out constant attributes
        if filter_columns_face is not None:
            faceAttrs = np.delete(faceAttrs, filter_columns_face, axis=1)
        if filter_columns_edge is not None:
            edgeAttrs = np.delete(edgeAttrs, filter_columns_edge, axis=1)
        faceAttrsScaler = StandardScaler().fit(faceAttrs)
        edgeAttrsScaler = StandardScaler().fit(edgeAttrs)
        return faceAttrsScaler, edgeAttrsScaler

    def get_vars_scaler(self, filter_columns=None):
        vars = np.array(self.vars)
        if filter_columns is not None:
            vars = np.delete(vars, filter_columns, axis=1)
        varsScaler = StandardScaler().fit(vars)
        return varsScaler

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

    def _collate(self, batch):
        batched_graph = Batch.from_data_list([x["graph"] for x in batch])
        batch_filename = [x["filename"] for x in batch]
        collated = {}
        collated["graph"] = batched_graph
        collated["filename"] = batch_filename
        collated["label"] =  torch.Tensor([x["label"] for x in batch])
        collated["vars"] =  torch.cat([x["vars"] for x in batch], dim=0)
        return collated
    
    def get_dataloader(self, batch_size=128, shuffle=True, num_workers=0, drop_last=True):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
            num_workers=num_workers,  # Can be set to non-zero on Linux
            drop_last=drop_last,
        )