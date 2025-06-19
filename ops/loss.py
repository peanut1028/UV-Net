#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   loss.py
@Time    :   2025/05/26 11:50:47
@Author  :   LGJ 
@Version :   1.0
@Contact :   lgjhsjt@163.com
@License :   (C)Copyright 2022-2025
@Desc    :   None
'''

# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveRelativeLoss(nn.Module):
    def __init__(self, alpha=0.05, beta=0.05, epsilon=1e-6):
        super().__init__()
        self.alpha = alpha       # 个体差异权重
        self.beta = beta         # 全局方差权重
        self.epsilon = epsilon   # 平滑项
        self.register_buffer('y_var', None)  # 训练集方差（不参与梯度计算）

    def set_y_var(self, y_train):
        """ 必须在训练前调用以设置方差 """
        if isinstance(y_train, torch.Tensor):
            self.y_var = torch.var(y_train)
        else:
            self.y_var = torch.var(torch.tensor(y_train))

    def forward(self, y_pred, y_true, reduction='mean'):
        if self.y_var is None:
            raise RuntimeError("Must call set_y_var() before using this loss!")
        
        absolute_error = torch.abs(y_pred - y_true)
        denominator = self.alpha * torch.abs(y_true) + self.beta * self.y_var + self.epsilon
        loss = absolute_error / denominator
        if reduction =='mean':
            loss = torch.mean(loss)
        if reduction =='sum':
            loss = torch.sum(loss)
        return loss