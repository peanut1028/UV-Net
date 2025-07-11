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

import pathlib
import time
import pandas as pd
import os
import json
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
import torch
from torch_geometric.utils import degree

from datasets.atwcad import ATWCADDataset
from datasets.atwmat import ATWMATDataset
from models.models import Regression


def main(accelerator="gpu",             # "cpu" or "gpu" or "tpu"
        devices=[0],                     # number of devices to use for training (only for GPU/TPU)
        max_epochs=1000,                # maximum number of epochs to train (only for training)
        check_val_every_n_epoch=5,     # check validation every n epochs (only for training)
        accumulate_grad_batches=2,     # number of batches to accumulate before performing an optimization step (only for training)
        amp_backend="native",          # mixed precision backend to use. Options: 'native', 'apex'
        auto_lr_find=True,             # whether to perform automatic learning rate finding (only for training)
        auto_scale_batch_size="power",    # whether to perform automatic scaling of the batch size (only for training)
        center_and_scale=False,         # whether to center and scale the data before training (only for training)
        log_every_n_steps=20,
        datasetDir=r"E:\Project\AutoPricing\datasets",
        checkpointPath=r"E:\LGJ\program\UV-Net\results\regression\0115\120043\epoch=884-val_loss=34.56-val_acc=0.72.ckpt",
        mode="train",
        dataset="atwcad",
        batch_size=256,
        num_workers=0,
        experiment_name="pna",
        init_lr=1e-2,
        num_classes=1,
        vars_dim=8,
        node_feat_dim=18,
        edge_feat_dim=18,
        hidden_channels=128,
        num_layers=3,
        lossfn='L1',
        scheduler=None,
        scaler_file=None
    ):
    datasetDir = os.path.join(datasetDir,dataset)
    results_path = (pathlib.Path(__file__).parent.joinpath("results").joinpath(experiment_name))
    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)

    # Define a path to save the results based date and time. E.g.
    # results/args.experiment_name/0430/123103
    month_day = time.strftime("%m%d")
    hour_min_second = time.strftime("%H%M%S")

    lr_callback = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        save_top_k=-1,
        dirpath=str(results_path.joinpath(month_day, hour_min_second)),
        filename='{epoch}-{val_loss:.4f}-{val_acc:.4f}',
        save_last=True,
    )

    trainer = Trainer(
        callbacks=[checkpoint_callback, lr_callback],
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
        log_every_n_steps=log_every_n_steps, 
    )

    if dataset == "atwcad":
        Dataset = ATWCADDataset
    elif dataset == "atwmaterial":
        Dataset = ATWMATDataset
    else:
        raise ValueError("Unsupported dataset")

    if mode == "train":
        # Train/val
        seed_everything(workers=True)
        print(
            f"""
    -----------------------------------------------------------------------------------
    PNA Regression on /{dataset} dataset
    -----------------------------------------------------------------------------------
    Logs written to results/{experiment_name}/{month_day}/{hour_min_second}

    To monitor the logs, run:
    tensorboard --logdir results/{experiment_name}/{month_day}/{hour_min_second}

    The trained model with the best validation loss will be written to:
    results/{experiment_name}/{month_day}/{hour_min_second}/best.ckpt
    -----------------------------------------------------------------------------------
        """
        )
        train_data = Dataset(root_dir=datasetDir, 
                            center_and_scale=center_and_scale, 
                            mode="train")
        val_data = Dataset(root_dir=datasetDir, 
                        center_and_scale=center_and_scale, 
                        mode="val")
        train_loader = train_data.get_dataloader(
            batch_size=batch_size, shuffle=True, num_workers=num_workers
        ) 
        val_loader = val_data.get_dataloader(
            batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
        )

        # calculate deg histogram
        deg = torch.zeros(5000, dtype=torch.long)
        for data in train_data:
            graph = data["graph"]
            d = degree(graph.edge_index[0], graph.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        
        model = Regression(num_classes=num_classes,
                           vars_dim=vars_dim,
                           node_feat_dim=node_feat_dim,
                           edge_feat_dim=edge_feat_dim,
                           hidden_channels=hidden_channels,
                           num_layers=num_layers,
                           deg=deg,
                           lossfn=lossfn,
                           init_lr=init_lr,
                           scheduler=scheduler,
                           scaler_file=os.path.join(datasetDir, scaler_file) if scaler_file else None
                           )
        trainer.fit(model, train_loader, val_loader)

    else:
        # Test
        assert (
            checkpointPath is not None
        ), "Expected the --checkpoint argument to be provided"
        test_data = Dataset(root_dir=datasetDir, mode="test")
        test_loader = test_data.get_dataloader(
            batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
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


if __name__ == "__main__":
    main()
