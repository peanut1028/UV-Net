#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   pricing.py
@Time    :   2024/09/06 17:31:34
@Author  :   LGJ 
@Version :   1.0
@Contact :   lgjhsjt@163.com
@License :   (C)Copyright 2022-2025
@Desc    :   None
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
from uvnet.models import Regression


# Define the arguments for the script
accelerator = "gpu"             # "cpu" or "gpu" or "tpu"
devices = 1                     # number of devices to use for training (only for GPU/TPU)
max_epochs = 500                # maximum number of epochs to train (only for training)
check_val_every_n_epoch = 5     # check validation every n epochs (only for training)
accumulate_grad_batches = 2     # number of batches to accumulate before performing an optimization step (only for training)
amp_backend = "native"          # mixed precision backend to use. Options: 'native', 'apex'
auto_lr_find = True             # whether to perform automatic learning rate finding (only for training)
auto_scale_batch_size = "power"    # whether to perform automatic scaling of the batch size (only for training)
use_swa = True                  # whether to use stochastic weight averaging (only for training)
use_CyclicLR = True             # whether to use cyclical learning rate (only for training)

parser = argparse.ArgumentParser("UV-Net solid model regression")
parser.add_argument(
    "--mode", 
    choices=("train", "test"), 
    default="test",
    help="Whether to train or test"
)
parser.add_argument("--dataset", 
                    choices=("atwcad",),
                    default="atwcad", 
                    help="Dataset to train on")
parser.add_argument("--dataset_path", type=str, 
                    default=r"E:\Project\AutoPricing\datasets\atwcad",
                    help="Path to dataset")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
parser.add_argument(
    "--num_workers",
    type=int,
    default=0,
    help="Number of workers for the dataloader. NOTE: set this to 0 on Windows, any other value leads to poor performance",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=r"E:\LGJ\program\UV-Net\results\regression\1008\142458\epoch=344-val_loss=97.76-val_acc=0.80.ckpt",
    help="Checkpoint file to load weights from for testing",
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default="regression",
    help="Experiment name (used to create folder inside ./results/ to save logs and checkpoints)",
)

parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

results_path = (
    pathlib.Path(__file__).parent.joinpath("results").joinpath(args.experiment_name)
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

trainer = Trainer.from_argparse_args(
    args,
    callbacks=[checkpoint_callback],
    logger=TensorBoardLogger(
        str(results_path), name=month_day, version=hour_min_second,
    ),
    accelerator=accelerator,
    devices=devices,
    max_epochs=max_epochs,
    check_val_every_n_epoch=check_val_every_n_epoch,
    accumulate_grad_batches=accumulate_grad_batches,
    amp_backend=amp_backend,
    auto_lr_find=auto_lr_find,
    auto_scale_batch_size=auto_scale_batch_size,  
    log_every_n_steps=23,  
)

if args.dataset == "atwcad":
    Dataset = ATWCADDataset
else:
    raise ValueError("Unsupported dataset")

if args.mode == "train":
    # Train/val
    seed_everything(workers=True)
    print(
        f"""
-----------------------------------------------------------------------------------
UV-Net Regression on atwcad pricing dataset
-----------------------------------------------------------------------------------
Logs written to results/{args.experiment_name}/{month_day}/{hour_min_second}

To monitor the logs, run:
tensorboard --logdir results/{args.experiment_name}/{month_day}/{hour_min_second}

The trained model with the best validation loss will be written to:
results/{args.experiment_name}/{month_day}/{hour_min_second}/best.ckpt
-----------------------------------------------------------------------------------
    """
    )
    model = Regression(num_classes=1)
    train_data = Dataset(root_dir=args.dataset_path, mode="train")
    val_data = Dataset(root_dir=args.dataset_path, mode="val")
    train_loader = train_data.get_dataloader(
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = val_data.get_dataloader(
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False
    )
    trainer.fit(model, train_loader, val_loader)
else:
    # Test
    assert (
        args.checkpoint is not None
    ), "Expected the --checkpoint argument to be provided"
    test_data = Dataset(root_dir=args.dataset_path, mode="test")
    test_loader = test_data.get_dataloader(
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False
    )
    model = Regression.load_from_checkpoint(args.checkpoint)
    results = trainer.predict(model=model, dataloaders=[test_loader])
    preds = torch.cat([x["preds"] for x in results])
    labels = torch.cat([x["labels"] for x in results])
    acc = 1 - (torch.abs(preds - labels) / labels).mean()
    print(
        f"Regression accuracy on test set: {acc:.4f}"
    )
    # write predictions to file
    df = pd.DataFrame(
        {"predicted_price": preds.numpy(), "actual_price": labels.numpy()}
    )
    df.to_csv(results_path.joinpath(f"test_results_{month_day}_{hour_min_second}_{acc:.4f}.csv"), index=False)
