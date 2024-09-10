import pathlib
import string
import os.path as osp
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from datasets.base import BaseDataset



class ATWCADDataset(BaseDataset):

    def __init__(
        self,
        root_dir,
        mode="train",
        center_and_scale=True,
        random_rotate=False,
    ):
        """
        Load the ATWCAD dataset

        ATWCAD
        |
        ├── train.txt
            ├── file1.bin  主材重量 主材单价 主材系数 主材费用 单价(gt)
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
            random_rotate (bool, optional): Whether to apply random rotations to the solid in 90 degree increments. Defaults to False.
        """
        assert mode in ("train", "val", "test")
        path = pathlib.Path(root_dir)

        self.random_rotate = random_rotate

        self.file_paths = []
        self.vars = []
        self.labels = []

        if mode == "train":
            with open(osp.join(root_dir, "train.txt"), "r") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == '':
                        continue
                    filename, annostr = line.rsplit('  ', 1)
                    values = [float(x) for x in annostr.split(' ')]
                    assert len(values) == 5, "{} has wrong number of values".format(filename)
                    self.file_paths.append(filename)
                    self.vars.append(values[:4])
                    self.labels.append(values[4])
        elif mode == "val":
            with open(osp.join(root_dir, "val.txt"), "r") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == '':
                        continue
                    filename, annostr = line.rsplit('  ', 1)
                    values = [float(x) for x in annostr.split(' ')]
                    assert len(values) == 5, "{} has wrong number of values".format(filename)
                    self.file_paths.append(filename)
                    self.vars.append(values[:4])
                    self.labels.append(values[4])
        elif mode == "test":
            with open(osp.join(root_dir, "test.txt"), "r") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == '':
                        continue
                    filename, annostr = line.rsplit('  ', 1)
                    values = [float(x) for x in annostr.split(' ')]
                    assert len(values) == 5, "{} has wrong number of values".format(filename)
                    self.file_paths.append(filename)
                    self.vars.append(values[:4])
                    self.labels.append(values[4])


        print(f"Loading {mode} data...")
        self.load_graphs(center_and_scale)
        print("Done loading {} files".format(len(self.data)))

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
        collated["label"] =  torch.cat([x["label"] for x in batch], dim=0)
        collated["vars"] =  torch.cat([x["vars"] for x in batch], dim=0)
        return collated
