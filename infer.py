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
from models.models import Regression


# MODEL_PATH = r"E:\Project\AutoPricing\sheet-metal-pricing\models\machining_pricing_model.ckpt"
MODEL_PATH = r"E:\Project\AutoPricing\sheet-metal-pricing\models\volume_model.ckpt"
TEST_FILE = r"E:\Project\AutoPricing\datasets\atwcad\pt\A059B-22-01-02-064A.pt"
TEST_VAR = [2.0949535, 5.0, 0, 2, 122, 13.08, 1.35]  


def load_test_data(filename, var, convertCenter=True):
    if not os.path.exists(filename):
        return None
    graph = torch.load(filename)
    sample = {"graph": graph, 
              "vars": torch.tensor(var, dtype=torch.float32)}
    return sample


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_sample = load_test_data(TEST_FILE, TEST_VAR)
graph = test_sample["graph"].to(device)
node_feat = graph.x.reshape(-1, 18).float()
edge_feat = graph.edge_attr.reshape(-1, 18).float()
vars = test_sample["vars"].unsqueeze(0).to(device)
batch = torch.zeros(graph.x.size(0), dtype=torch.long).to(device)
model = Regression.load_from_checkpoint(MODEL_PATH).to(device)
model.eval()
with torch.no_grad():
    result = model.model(node_feat, graph.edge_index, edge_feat, vars, batch)
print(f"The predicted price of file {TEST_FILE} is {result.item()}")
