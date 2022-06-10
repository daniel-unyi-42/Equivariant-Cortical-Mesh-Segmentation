import torch

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

import numpy as np

class HCPDataset(InMemoryDataset):
    
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['Label_data_in_mask_100.txt', 'edges.txt', 'feature_matrix_100.npy']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        raise RuntimeError(
            f"Dataset not found. Move it to '{self.raw_dir}'")

    def process(self):
        xs = np.load(self.raw_dir + "/feature_matrix_100.npy").astype(np.float32)
        xs, poss = xs[:,:,:9], xs[:,:,9:]
        edge_index = np.loadtxt(self.raw_dir + "/edges.txt", dtype = np.int64).T
        edge_index = np.concatenate([edge_index[[1, 0]], edge_index[[0, 1]]], axis=1) # to undirected
        ys = np.loadtxt(self.raw_dir + "/Label_data_in_mask_100.txt", dtype=np.int64)
        xs = torch.from_numpy(xs) # node features
        poss = torch.from_numpy(poss) # node positions
        edge_index = torch.from_numpy(edge_index) # edge structure of each mesh
        ys = torch.from_numpy(ys) # node labels
        data_list = [Data(x=x, pos=pos, edge_index=edge_index, y=y)
                     for x, pos, y in zip(xs, poss, ys)]
        #np.random.shuffle(data_list)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
