#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   infer.py
@Time    :   2024/11/08 15:24:09
@Author  :   LGJ 
@Version :   1.0
@Contact :   lgjhsjt@163.com
@License :   (C)Copyright 2022-2025
@Desc    :   infer single sample
'''

# here put the import lib
import os
import torch
from dgl.data.utils import load_graphs
from uvnet.models import Regression
from datasets import util


MODEL_PATH = r"E:\LGJ\program\UV-Net\results\regression\0211\182928\epoch=439-val_loss=3.73-val_acc=0.82.ckpt"
TEST_FILE = r"E:\Project\AutoPricing\datasets\atwcad\bin\B018F-43-01-015A.bin"
TEST_VAR = [577, 799.57, 15767.1, 1969, 1, 4172, 0.0]

def center_and_scale(graph):
    graph.ndata["x"], center, scale = util.center_and_scale_uvgrid(
        graph.ndata["x"], return_center_scale=True
    )
    graph.edata["x"][..., :3] -= center
    graph.edata["x"][..., :3] *= scale
    return graph

def load_test_data(filename, var, convertCenter=True):
    if not os.path.exists(filename):
        return None
    graph = load_graphs(str(filename))[0][0]
    if convertCenter:
        graph = center_and_scale(graph)
    graph.ndata["x"] = graph.ndata["x"].type(torch.FloatTensor)
    graph.edata["x"] = graph.edata["x"].type(torch.FloatTensor)
    sample = {"graph": graph, 
              "vars": torch.tensor(var, dtype=torch.float32)}
    return sample


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_sample = load_test_data(TEST_FILE, TEST_VAR)
graph = test_sample["graph"].to(device)
graph.ndata["x"] = graph.ndata["x"].permute(0, 3, 1, 2)
graph.edata["x"] = graph.edata["x"].permute(0, 2, 1)
vars = test_sample["vars"].unsqueeze(0).to(device)

model = Regression()
model = model.load_from_checkpoint(MODEL_PATH).to(device)
model.eval()
with torch.no_grad():
    result = model.model(graph, vars)
print(f"The predicted price of file {TEST_FILE} is {result.item()}")
