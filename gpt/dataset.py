import torch
import uproot
import numpy as np

from torch.utils.data import Dataset

class PairedData(Dataset):
    """
    Paired data for training the GPT model
    """
    def __init__(self, 
                 data_path='/global/cfs/cdirs/dune/users/rradev/near_to_far/paired_ndfd.root', 
                 near_reco=None, 
                 far_reco=None,
                 train=True):
        
        super().__init__()
        self.data_path = data_path
        self.train = train
       
        if near_reco is None:
            near_reco = [
                'eRecoP', 'eRecoN', 'eRecoPip', 
                'eRecoPim', 'eRecoPi0', 'eRecoOther', 
                'Ev_reco', 'Elep_reco', 'theta_reco'
            ]

        if far_reco is None:
            far_reco = [
                'nue_nu_E',
                'numu_nu_E',
                'nc_nu_E',
            ]
    
        self.near_reco = near_reco
        self.far_reco = far_reco

        self.data = self.load_data()
        self.block_size = len(near_reco) + len(far_reco) + 1 

    def load_data(self):
        tree = uproot.open(self.data_path)['nd_fd_reco']

        near_det = tree.arrays(self.near_reco, library='pd').to_numpy()
        far_det = tree.arrays(self.far_reco, library='pd').to_numpy()
        
        data = np.concatenate((near_det, far_det), axis=1)
        
        samples_in_train = 45_000
       
        if self.train:
            self.near_data = np.log1p(near_det[:samples_in_train])
            self.far_data = np.log1p(far_det[:samples_in_train])
            data = data[:samples_in_train]
        
        else:
            self.near_data = np.log1p(near_det[samples_in_train:])
            self.far_data = np.log1p(far_det[samples_in_train:])
            data = data[samples_in_train:]
            
        return np.log1p(data)

    def get_block_size(self):
        return self.block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # return as tensors
        x = torch.tensor(sample[:-1], dtype=torch.float)
        y = torch.tensor(sample[1:], dtype=torch.float)
        y[:len(self.near_reco) - 1] = -1 #only compute loss far det

        return x, y


# if name main
if __name__ == "__main__":
    dataset = PairedData('/global/cfs/cdirs/dune/users/rradev/near_to_far/paired_ndfd.root')
    print(dataset[0])
    print(dataset[0][0].shape)