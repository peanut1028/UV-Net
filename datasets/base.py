import torch
from torch.utils.data import Dataset, DataLoader
from torch import FloatTensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader
from datasets import util
from tqdm import tqdm
from abc import abstractmethod


class BaseDataset(Dataset):
    @staticmethod
    @abstractmethod
    def num_classes():
        pass

    def load_graphs(self, file_paths, center_and_scale=True):
        self.data = []
        for fn in tqdm(file_paths):
            if not fn.exists():
                continue
            sample = self.load_one_graph(fn)
            if sample is None:
                continue
            if sample["graph"].edata["x"].size(0) == 0:
                # Catch the case of graphs with no edges
                continue
            self.data.append(sample)
        self.convert_to_float32()
    
    def load_one_graph(self, file_path):
        graph = torch.load(file_path)
        sample = {"graph": graph, "filename": file_path.stem}
        return sample

    def convert_to_float32(self):
        for i in range(len(self.data)):
            self.data[i]["graph"].x = self.data[i]["graph"].x.to(torch.float32)
            self.data[i]["graph"].edge_attr = self.data[i]["graph"].edge_attr.to(torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

    def get_dataloader(self, batch_size=128, shuffle=True, num_workers=0, drop_last=True):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
            num_workers=num_workers,  # Can be set to non-zero on Linux
            drop_last=drop_last,
        )
