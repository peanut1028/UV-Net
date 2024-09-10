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

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

from datasets.atwcad import ATWCADDataset
from uvnet.models import Regression

parser = argparse.ArgumentParser("UV-Net solid model regression")
parser.add_argument(
    "traintest", choices=("train", "test"), help="Whether to train or test"
)
parser.add_argument("--dataset", choices=("atwcad",), help="Dataset to train on")
parser.add_argument("--dataset_path", type=str, help="Path to dataset")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument(
    "--num_workers",
    type=int,
    default=0,
    help="Number of workers for the dataloader. NOTE: set this to 0 on Windows, any other value leads to poor performance",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint file to load weights from for testing",
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default="classification",
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
    monitor="val_loss",
    dirpath=str(results_path.joinpath(month_day, hour_min_second)),
    filename="best",
    save_last=True,
)

trainer = Trainer.from_argparse_args(
    args,
    callbacks=[checkpoint_callback],
    logger=TensorBoardLogger(
        str(results_path), name=month_day, version=hour_min_second,
    ),
)

if args.dataset == "atwcad":
    Dataset = ATWCADDataset
else:
    raise ValueError("Unsupported dataset")

if args.traintest == "train":
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
    train_data = Dataset(root_dir=args.dataset_path, split="train")
    val_data = Dataset(root_dir=args.dataset_path, split="val")
    train_loader = train_data.get_dataloader(
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = val_data.get_dataloader(
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    trainer.fit(model, train_loader, val_loader)
else:
    # Test
    assert (
        args.checkpoint is not None
    ), "Expected the --checkpoint argument to be provided"
    test_data = Dataset(root_dir=args.dataset_path, split="test")
    test_loader = test_data.get_dataloader(
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    model = Regression.load_from_checkpoint(args.checkpoint)
    results = trainer.test(model=model, test_dataloaders=[test_loader], verbose=False)
    print(
        f"Regression mae on test set: {results[0]['test_acc_epoch']:.4f}"
    )
