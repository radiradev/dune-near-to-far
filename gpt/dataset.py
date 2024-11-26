import torch
import uproot
import numpy as np
from torch.utils.data import Dataset
import pandas as pd


class NewPairedData(Dataset):
    """
    Paired data for training the GPT model
    """
    def __init__(self,
                 data_path='data/ndfd_reco_only_cuts.noFDhasel_oldg4params.csv',
                 near_reco=None,
                 far_reco=None,
                 train=True,
                 sample_weight_var=None,
                 resample_data=None):

        super().__init__()
        self.data_path = data_path
        self.train = train
        self.sample_weight_var = [] if sample_weight_var is None else [sample_weight_var]
        if resample_data is not None:
            self.resample_bins = resample_data[0]
            self.resample_probs = resample_data[1]
            self.sample_weight_var = [resample_data[2]]
            self.resample_min = resample_data[3]
            self.resample_max = resample_data[4]
            self.resample_binl_idxs = np.where(self.resample_probs)[0]
            self.resample = True
        else:
            self.resample = False

        if near_reco is None:
            # -- default
            # near_reco = [
            #     'eRecoP', 'eRecoN', 'eRecoPip',
            #     'eRecoPim', 'eRecoPi0', 'eRecoOther',
            #     'Ev_reco', 'Elep_reco', 'theta_reco',
            #     'reco_numu', 'reco_nc', 'reco_nue', 'reco_lepton_pdg',
            #     'fd_x_vert', 'fd_y_vert', 'fd_z_vert',
            # ]
            # -- noN_inputnparticles
            # near_reco = [
            #     'eRecoP', 'eRecoPip', 'eRecoPim', 'eRecoPi0', 'eRecoOther',
            #     'nP', 'nipip', 'nipim', 'nipi0', 'nipi0', 'nikp', 'nikm', 'nik0', 'niem', 'niother',
            #     'Ev_reco', 'Elep_reco', 'theta_reco',
            #     'reco_numu', 'reco_nc', 'reco_nue', 'reco_lepton_pdg',
            #     'fd_x_vert', 'fd_y_vert', 'fd_z_vert',
            # ]
            # -- noN
            # near_reco = [
            #     'eRecoP', 'eRecoPip', 'eRecoPim', 'eRecoPi0', 'eRecoOther',
            #     'Ev_reco', 'Elep_reco', 'theta_reco',
            #     'reco_numu', 'reco_nc', 'reco_nue', 'reco_lepton_pdg',
            #     'fd_x_vert', 'fd_y_vert', 'fd_z_vert',
            # ]
            # -- noN_noleppdg # NOTE I accidently left 'reco_lepton_pdg' in here for some experiments
            # near_reco = [
            #     'eRecoP', 'eRecoPip', 'eRecoPim', 'eRecoPi0', 'eRecoOther',
            #     'Ev_reco', 'Elep_reco', 'theta_reco',
            #     'reco_numu', 'reco_nc', 'reco_nue',
            #     'fd_x_vert', 'fd_y_vert', 'fd_z_vert',
            # ]
            # -- noN_sensible
            # near_reco = [
            #     'nP', 'nipipm', 'nikpm', 'nipi0', 'nik0', 'niem', 'niother',
            #     'eRecoP', 'eRecoPipm', 'eRecoPi0', 'eRecoOther',
            #     'Ev_reco', 'Elep_reco', 'theta_reco',
            #     'muon_tracker', 'muon_contained', 'Ehad_veto',
            #     'fd_x_vert', 'fd_y_vert', 'fd_z_vert',
            # ]
            # -- noN_sensible2
            # near_reco = [
            #     'eRecoP', 'eRecoPipm', 'eRecoPi0', 'eRecoOther',
            #     'Ev_reco', 'Elep_reco', 'theta_reco',
            #     'muon_tracker', 'muon_contained', 'Ehad_veto',
            #     'fd_x_vert', 'fd_y_vert', 'fd_z_vert',
            # ]
            # -- noN_sensible3
            near_reco = [
                'nP', 'nipipm', 'nikpm', 'nipi0', 'nik0', 'niem', 'niother',
                'eRecoP', 'eRecoPipm', 'eRecoPi0', 'eRecoOther',
                'Ev_reco', 'Elep_reco', 'theta_reco',
                'muon_tracker', 'muon_contained', 'Ehad_veto',
                'fd_x_vert_fv_mindist',
                'fd_y_vert_fv_mindist',
                'fd_z_vert_fv_frontdist', 'fd_z_vert_fv_backdist'
            ]
            # -- noN_trackercontained
            # near_reco = [
            #     'eRecoP', 'eRecoPip', 'eRecoPim', 'eRecoPi0', 'eRecoOther',
            #     'Ev_reco', 'Elep_reco', 'theta_reco',
            #     'muon_tracker', 'muon_contained',
            #     'reco_numu', 'reco_nc', 'reco_nue', 'reco_lepton_pdg',
            #     'fd_x_vert', 'fd_y_vert', 'fd_z_vert',
            # ]
            # -- noN_trackercontained_ehadveto
            # near_reco = [
            #     'eRecoP', 'eRecoPip', 'eRecoPim', 'eRecoPi0', 'eRecoOther',
            #     'Ev_reco', 'Elep_reco', 'theta_reco',
            #     'muon_tracker', 'muon_contained', 'Ehad_veto',
            #     'reco_numu', 'reco_nc', 'reco_nue',
            #     'fd_x_vert', 'fd_y_vert', 'fd_z_vert',
            # ]
            # -- noN_trackercontained_ehadveto_inputparticles
            # near_reco = [
            #     'eRecoP', 'eRecoPip', 'eRecoPim', 'eRecoPi0', 'eRecoOther',
            #     'nP', 'nipip', 'nipim', 'nipi0', 'nipi0', 'nikp', 'nikm', 'nik0', 'niem', 'niother',
            #     'Ev_reco', 'Elep_reco', 'theta_reco',
            #     'muon_tracker', 'muon_contained', 'Ehad_veto',
            #     'reco_numu', 'reco_nc', 'reco_nue',
            #     'fd_x_vert', 'fd_y_vert', 'fd_z_vert',
            # ]

        if far_reco is None:
            # -- default
            # cvn_scores = ['fd_numu_score']#, 'fd_nue_score', 'fd_nc_score', 'fd_nutau_score']
            # far_reco = ['fd_numu_nu_E', 'fd_numu_lep_E', 'fd_numu_had_E'] #,'fd_nue_lep_E', 'fd_numu_nu_E', 'fd_nue_nu_E']
            # -- allcvn
            # cvn_scores = ['fd_numu_score', 'fd_nue_score', 'fd_nc_score', 'fd_nutau_score']
            # far_reco = ['fd_numu_nu_E', 'fd_numu_lep_E', 'fd_numu_had_E']
            # -- 1cvn
            cvn_scores = ['fd_numu_score']
            far_reco = ['fd_numu_nu_E', 'fd_numu_lep_E', 'fd_numu_had_E']
            # -- allcvn_fdnuElast
            # cvn_scores = ['fd_numu_score', 'fd_nue_score', 'fd_nc_score', 'fd_nutau_score']
            # far_reco = ['fd_numu_lep_E', 'fd_numu_had_E', 'fd_numu_nu_E']
            # -- allcvn_fdlepinfo
            # cvn_scores = ['fd_numu_score', 'fd_nue_score', 'fd_nc_score', 'fd_nutau_score']
            # far_reco = ['fd_numu_nu_E', 'fd_numu_lep_E', 'fd_numu_reco_method', 'fd_numu_had_E']
            # -- mostcvn
            # cvn_scores = ['fd_numu_score', 'fd_nue_score', 'fd_nc_score']
            # far_reco = ['fd_numu_nu_E', 'fd_numu_lep_E', 'fd_numu_had_E']
            # --allcvn_reordered
            # cvn_scores = ['fd_nue_score', 'fd_nc_score', 'fd_nutau_score', 'fd_numu_score']
            # far_reco = ['fd_numu_lep_E', 'fd_numu_had_E', 'fd_numu_nu_E']
            # -- minimal
            # cvn_scores = ['fd_numu_score']
            # far_reco = ['fd_numu_nu_E']

        self.cvn_scores = cvn_scores
        self.near_reco = near_reco
        self.far_reco = far_reco
        self.data = self.load_data()

        self.block_size = len(near_reco) + len(cvn_scores) + len(far_reco) + 1

        if self.resample:
            self.resample_vars, self.resample_vars_sorted, self.idxs_sorted = self._get_idx_resample_vars()
            self.data = self.data[:, :-1] # Get rid of the sample weight column now

    def load_data(self):
        df = pd.read_csv(self.data_path)
        # load in the near reco and far reco columsn

        df = df[self.near_reco + self.cvn_scores + self.far_reco + self.sample_weight_var]
        data = df.to_numpy().astype(np.float32)
        samples_in_train = 70_000

        if self.train:
            data = data[:samples_in_train]
        else:
            data = data[samples_in_train:]

        return data

    def _get_idx_resample_vars(self):
        resample_vars = np.array(self.data[:, -1])
        idxs = np.argsort(resample_vars)
        resample_vars_sorted = resample_vars[idxs]

        return resample_vars, resample_vars_sorted, idxs

    def get_scores_length(self):
        return len(self.cvn_scores)

    def get_near_reco_length(self):
        return len(self.near_reco)

    def get_far_reco_length(self):
        return len(self.far_reco)

    def get_block_size(self):
        return self.block_size

    def _resample(self):
        binl_idx = np.random.choice(self.resample_binl_idxs, p=self.resample_probs)
        resample_var = np.random.uniform(
            self.resample_bins[binl_idx], self.resample_bins[binl_idx + 1]
        )
        idx = np.searchsorted(self.resample_vars_sorted, resample_var, side="left")
        if (
            idx > 0 and
            (
                idx == len(self.resample_vars_sorted) or
                (
                    abs(resample_var - self.resample_vars_sorted[idx - 1]) <
                    abs(resample_var - self.resample_vars_sorted[idx])
                )
            )
        ):
            return self.idxs_sorted[idx - 1]
        else:
            return self.idxs_sorted[idx]

    def _uniform_sample_Ev(self):
        Ev = np.random.uniform(0.5, 6.0)
        idx = np.searchsorted(self.Evs_sorted, Ev, side="left")
        if (
            idx > 0 and
            (
                idx == len(self.Evs_sorted) or
                abs(Ev - self.Evs_sorted[idx - 1]) < abs(Ev - self.Evs_sorted[idx])
            )
        ):
            return self.idxs_sorted[idx - 1]
        else:
            return self.idxs_sorted[idx]

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        if self.resample:
            if (
                self.resample_vars[idx] < self.resample_min or
                self.resample_vars[idx] > self.resample_max
            ):
                new_idx = idx
            else:
                new_idx = self._resample()
            sample = torch.tensor(self.data[new_idx], dtype=torch.float)
            return sample[:-1], sample[len(self.near_reco):]

        sample = torch.tensor(self.data[idx], dtype=torch.float)
        if not self.sample_weight_var:
            return sample[:-1], sample[len(self.near_reco):]
        else:
            sample_weight_var_val = sample[-1]
            sample = sample[:-1]
            return sample[:-1], sample[len(self.near_reco):],  sample_weight_var_val


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
                'reco_numu', 'reco_nc', 'reco_nue', 'reco_lepton_pdg',
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
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float)
        return sample[:-1], sample[len(self.near_reco):]

# if name main
if __name__ == "__main__":
    dataset = PairedData('/global/cfs/cdirs/dune/users/rradev/near_to_far/paired_ndfd.root')
    print(dataset[0])
    print(dataset[0][0].shape)

