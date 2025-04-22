#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   export.py
@Time    :   2025/01/09 18:30:28
@Author  :   LGJ 
@Version :   1.0
@Contact :   lgjhsjt@163.com
@License :   (C)Copyright 2022-2025
@Desc    :   export onnx model (not support yet)
'''

# here put the import lib
import os
import torch
from dgl.data.utils import load_graphs
from uvnet.models import Regression
from datasets import util



MODEL_PATH = r"results\regression\0103\165354\epoch=999-val_loss=62.33-val_acc=0.81.ckpt"
EXPORT_PATH = r"results\regression\0103\165354\model.onnx"
SAMPLE_INPUT_FILE = r"E:\Project\AutoPricing\datasets\atwcad\bin\A050EG-12-03-024A.bin"
TEST_VAR = [1365.0, 0.02, 17.7, 1.0, 0.4, 0.0, 0.0, 2.0, 1.0, 1.0]

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
test_sample = load_test_data(SAMPLE_INPUT_FILE, TEST_VAR)
graph = test_sample["graph"].to(device)
graph.ndata["x"] = graph.ndata["x"].permute(0, 3, 1, 2)
graph.edata["x"] = graph.edata["x"].permute(0, 2, 1)
vars = test_sample["vars"].unsqueeze(0).to(device)

model = Regression()
model = model.load_from_checkpoint(MODEL_PATH).to(device)
model.eval()

model.to_onnx(EXPORT_PATH, 
              input_sample=(graph, vars),
                )