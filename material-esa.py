#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   material.py
@Time    :   2024/09/06 17:31:34
@Author  :   LGJ 
@Version :   1.0
@Contact :   lgjhsjt@163.com
@License :   (C)Copyright 2022-2025
@Desc    :   Train/validate model to predict raw material weight of a step sheet metal part 
'''

# here put the import lib

import argparse
import pathlib
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
import torch

from datasets.atwcad import ATWCADDataset
from datasets.atwmat import ATWMATDataset
from esa.models import Regression


# ****************************************trainer hyperparameters**********************************************
train_args = dict(
                mode = "train",                   # "train" or "test"
                dataset = "atwmaterial",           # "atwcad" or "atwmaterial"
                batch_size = 256,                # batch size for training and validation
                num_workers = 0,                 # number of workers for the dataloader (set to 0 on Windows)
                experiment_name = "regression",   # experiment name (used to create folder inside ./results/ to save logs and checkpoints)
                accelerator = "gpu",          # "cpu" or "gpu" or "tpu"
                devices = [0],                   # number of devices to use for training (only for GPU/TPU)
                max_epochs = 500,              # maximum number of epochs to train (only for training)
                check_val_every_n_epoch = 5,   # check validation every n epochs (only for training)
                accumulate_grad_batches = 2,    # number of batches to accumulate before performing an optimization step (only for training)
                amp_backend = "native",        # mixed precision backend to use. Options: 'native', 'apex'
                auto_lr_find = True,           # whether to perform automatic learning rate finding (only for training)
                auto_scale_batch_size = "power",    # whether to perform automatic scaling of the batch size (only for training)
                use_swa = True,               # whether to use stochastic weight averaging (only for training)
                use_CyclicLR = True,           # whether to use cyclical learning rate (only for training)
                log_every_n_steps = 10
)

# ******************************************dataset hyperparameters*******************************************
center_and_scale = False         # whether to center and scale the data before training (only for training)
datasetDir = r"E:\Project\AutoPricing\datasets\atwmaterial"
checkpointPath = r"E:\LGJ\program\UV-Net\results\regression\0115\120043\epoch=884-val_loss=34.56-val_acc=0.72.ckpt"

# ******************************************model hyperparameters**********************************************
model_args = dict(
                num_features = 3,
                edge_dim = 3,
                xformers_or_torch_attn = "xformers",
                hidden_dims = [256, 256, 256, 256, 256, 256],
                num_heads = [16, 16, 16, 16, 16, 16],
                sab_dropout = 0.0,
                mab_dropout = 0.0,
                pma_dropout = 0.0,
                use_mlps = True,
                mlp_hidden_size = 256,
                mlp_type = "standard",
                norm_type = "BN",
                layer_types = ["M", "S", "M", "S", "M", "P"],
                attn_residual_dropout = 0.0,
                set_max_items = max_edge_global,  # TODO how to get this value?
                num_mlp_layers = 2,
                use_bfloat16 = True,
                pre_or_post = "post",
                pma_residual_dropout = 0.0,
                use_mlp_ln = True,
                mlp_dropout = 0.0,
                linear_output_size = 1
)

# *************************************************************************************************************
results_path = (
    pathlib.Path(__file__).parent.joinpath("results").joinpath(train_args["experiment_name"])
)
if not results_path.exists():
    results_path.mkdir(parents=True, exist_ok=True)

# Define a path to save the results based date and time. E.g.
# results/args.experiment_name/0430/123103
month_day = time.strftime("%m%d")
hour_min_second = time.strftime("%H%M%S")
checkpoint_callback = ModelCheckpoint(
    monitor="val_acc",
    save_top_k=-1,
    dirpath=str(results_path.joinpath(month_day, hour_min_second)),
    filename='{epoch}-{val_loss:.2f}-{val_acc:.2f}',
    save_last=True,
)

trainer = Trainer(
    **train_args,
    callbacks=[checkpoint_callback],
    logger=TensorBoardLogger(
        str(results_path), name=month_day, version=hour_min_second,
    ),
)

if train_args["dataset"] == "atwcad":
    Dataset = ATWCADDataset
elif train_args["dataset"] == "atwmaterial":
    Dataset = ATWMATDataset
else:
    raise ValueError("Unsupported dataset")

if train_args["mode"] == "train":
    # Train/val
    seed_everything(workers=True)
    print(
        f"""
-----------------------------------------------------------------------------------
UV-Net Regression on atwcad pricing dataset
-----------------------------------------------------------------------------------
Logs written to results/{train_args["experiment_name"]}/{month_day}/{hour_min_second}

To monitor the logs, run:
tensorboard --logdir results/{train_args["experiment_name"]}/{month_day}/{hour_min_second}

The trained model with the best validation loss will be written to:
results/{train_args["experiment_name"]}/{month_day}/{hour_min_second}/best.ckpt
-----------------------------------------------------------------------------------
    """
    )
    model = Regression(model_args)
    train_data = Dataset(root_dir=datasetDir, 
                         center_and_scale=center_and_scale, 
                         mode="train")
    val_data = Dataset(root_dir=datasetDir, 
                       center_and_scale=center_and_scale, 
                       mode="val")
    train_loader = train_data.get_dataloader(
        batch_size=train_args["batch_size"], shuffle=True, num_workers=train_args["num_workers"]
    ) 
    val_loader = val_data.get_dataloader(
        batch_size=train_args["batch_size"], shuffle=False, num_workers=train_args["num_workers"], drop_last=False
    )
    trainer.fit(model, train_loader, val_loader)
else:
    # Test
    assert (
        checkpointPath is not None
    ), "Expected the --checkpoint argument to be provided"
    test_data = Dataset(root_dir=datasetDir, mode="test")
    test_loader = test_data.get_dataloader(
        batch_size=train_args["batch_size"], shuffle=False, num_workers=train_args["num_workers"], drop_last=False
    )
    model = Regression.load_from_checkpoint(checkpointPath)
    results = trainer.predict(model=model, dataloaders=[test_loader])
    preds = torch.cat([x["preds"] for x in results])
    labels = torch.cat([x["labels"] for x in results])
    acc = 1 - (torch.abs(preds - labels) / labels).mean()
    print(
        f"Regression accuracy on test set: {acc:.4f}"
    )
    # write predictions to file
    with open(test_data.data_txt, "r") as f:
        lines = f.readlines()
    codes = []
    for l in lines:
        code, annostr = l.strip().rsplit("  ", 1)
        codes.append(code)

    df = pd.DataFrame(
        {"code": codes, "predicted_volume": preds.numpy(), "actual_volume": labels.numpy()}
    )
    df.to_csv(results_path.joinpath(f"test_results_{month_day}_{hour_min_second}_{acc:.4f}.csv"), index=False)
