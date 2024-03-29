import torch
import uproot
import numpy as np

from sklearn.preprocessing import PowerTransformer, QuantileTransformer
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
                'Ev_reco', 'Elep_reco', 'theta_reco',
                'reco_numu', 'reco_nc', 'reco_nue', 'reco_lepton_pdg'
            ]

        if far_reco is None:
            cvn_scores = ['numu_score', 'nue_score', 'nc_score', 'nutau_score']
            far_reco = ['nc_nu_E', 'numu_nu_E', 'nue_nu_E']
        
        self.cvn_scores = cvn_scores
        self.near_reco = near_reco
        self.far_reco = far_reco
        self.data = self.load_data()

        self.block_size = len(near_reco) + len(far_reco) + len(cvn_scores) + 1 

    def load_data(self):
        tree = uproot.open(self.data_path)['nd_fd_reco']

        near_det = tree.arrays(self.near_reco, library='pd').to_numpy()
        cvn_scores = tree.arrays(self.cvn_scores, library='pd').to_numpy()
        far_det = tree.arrays(self.far_reco, library='pd').to_numpy()
        # find rows where the energy is 0
        row_mask = far_det[:, 0] != 0
        
        data = np.concatenate((near_det, cvn_scores, far_det), axis=1)
        data = data[row_mask]
        samples_in_train = 70_000
       
        if self.train:
            self.near_data = near_det[:samples_in_train]
            self.far_data = far_det[:samples_in_train]
            data = data[:samples_in_train]
        
        else:
            self.near_data = near_det[samples_in_train:]
            self.far_data =  far_det[samples_in_train:]
            data = data[samples_in_train:]

        return data

    def get_scores_length(self):
        return len(self.cvn_scores)
    
    def get_far_reco_length(self):
        return len(self.far_reco)

    def get_block_size(self):
        return self.block_size

    def __len__(self):
        return len(self.near_data) - self.block_size

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float)
        return sample[:-1], sample[len(self.near_reco):]
    



# if name main
if __name__ == "__main__":
    dataset = PairedData('/global/cfs/cdirs/dune/users/rradev/near_to_far/paired_ndfd.root')
    print(dataset[0])
    print(dataset[0][0].shape)