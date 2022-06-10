import numpy as np
from scipy.sparse import coo_matrix
from scipy.stats import ortho_group
import torch
from torch.utils.data import Subset


# connects all neighbors within K hops
def connect_nodes(data, K=4):
    dummy_data = np.ones(len(data.edge_index.T))
    row, col = data.edge_index
    N = data.edge_index.max() + 1
    A = coo_matrix((dummy_data, (row, col)), shape=(N, N))
    A_Ks = [A.tocsr()]
    for k in range(K - 1):
        A_Ks.append(A_Ks[k].dot(A))
    A_K = sum(A_Ks)
    A_K.setdiag(0)
    A_K.eliminate_zeros()
    edge_index = np.stack(A_K.nonzero())
    data.edge_index = torch.Tensor(edge_index).long()
    return data


# global_=True: apply the same random isometric transformation to all meshes
# global_=False: apply different random isometric transformations to the different meshes
class RandomIsoTransform:
    
    def __init__(self, global_=False):
        self.global_ = global_
        if self.global_:
            self.ortho = torch.from_numpy(ortho_group.rvs(3)).to(torch.float32)
            self.translation = 2.0 * torch.rand((3)) - 1.0 # in the range [-1, 1]
            
    def __call__(self, data):
        if self.global_:
            data.pos = torch.mm(data.pos, self.ortho) + self.translation
        else:
            ortho = torch.from_numpy(ortho_group.rvs(3)).to(torch.float32)
            translation = 2.0 * torch.rand((3)) - 1.0 # in the range [-1, 1]
            data.pos = torch.mm(data.pos, ortho) + translation
        return data
    
    
def align_meshes(data_list):
    def align(pos1, pos2):
        for i in range(20):
            # centre of mass
            com1 = np.mean(pos1, axis=0)
            com2 = np.mean(pos2, axis=0)
            # SVD
            W = np.dot((pos1 - com1).T, (pos2 - com2))
            u, s, vh = np.linalg.svd(W)
            # result
            R = u.dot(vh) # orthogonal matrix
            r = com2 - com1.dot(R) # translation
            pos2 = pos2.dot(R.T) - r
        return pos2
    ref_mesh = data_list[0] # (arbitrary) reference mesh
    aligned_data_list = []
    for i, _ in enumerate(data_list):
        target_mesh = data_list[i] # target mesh
        target_pos = align(ref_mesh.pos.numpy(), target_mesh.pos.numpy())
        target_mesh.pos = torch.from_numpy(target_pos)
        aligned_data_list.append(target_mesh)
    return aligned_data_list


# splits the dataset into train, validation & test subsets, according to the actual fold
def split_fold(dataset, fold, number_of_folds):
    assert 0 <= fold <= number_of_folds - 1
    len_fold = len(dataset) // number_of_folds
    # indices for train, validation & test subsets
    test_low = (fold + 0) * len_fold % len(dataset)
    test_high = (fold + 1) * len_fold % len(dataset)
    if test_high < test_low:
        test_high = len(dataset)
    val_low = (fold + 1) * len_fold % len(dataset)
    val_high = (fold + 2) * len_fold % len(dataset)
    if val_high < val_low:
        val_high = len(dataset)
    all_indices = np.arange(len(dataset))
    np.random.shuffle(all_indices)
    test_indices = np.arange(test_low, test_high)
    train_indices = np.setdiff1d(all_indices, test_indices)
    val_indices = np.arange(val_low, val_high)
    train_indices = np.setdiff1d(train_indices, val_indices)
    # train, validation & test subsets of the dataset
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)
    return train_subset, val_subset, test_subset
