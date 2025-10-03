import torch
import uproot
import numpy as np
from torch.utils.data import Dataset
import pandas as pd


class NewPairedData(Dataset):
    """
    Paired data for training the GPT model
    """
    NEAR_RECO_PRESETS = {
        "default" :  [
            'eRecoP', 'eRecoN', 'eRecoPip',
            'eRecoPim', 'eRecoPi0', 'eRecoOther',
            'Ev_reco', 'Elep_reco', 'theta_reco',
            'reco_numu', 'reco_nc', 'reco_nue', 'reco_lepton_pdg',
            'fd_x_vert', 'fd_y_vert', 'fd_z_vert',
         ],
        "noN_inputnparticles" : [
            'eRecoP', 'eRecoPip', 'eRecoPim', 'eRecoPi0', 'eRecoOther',
            'nP', 'nipip', 'nipim', 'nipi0', 'nipi0', 'nikp', 'nikm', 'nik0', 'niem', 'niother',
            'Ev_reco', 'Elep_reco', 'theta_reco',
            'reco_numu', 'reco_nc', 'reco_nue', 'reco_lepton_pdg',
            'fd_x_vert', 'fd_y_vert', 'fd_z_vert',
        ],
        "noN" : [
            'eRecoP', 'eRecoPip', 'eRecoPim', 'eRecoPi0', 'eRecoOther',
            'Ev_reco', 'Elep_reco', 'theta_reco',
            'reco_numu', 'reco_nc', 'reco_nue', 'reco_lepton_pdg',
            'fd_x_vert', 'fd_y_vert', 'fd_z_vert',
        ],
        "noN_noleppdg" : [ # NOTE I accidently left 'reco_lepton_pdg' in here for some experiments
            'eRecoP', 'eRecoPip', 'eRecoPim', 'eRecoPi0', 'eRecoOther',
            'Ev_reco', 'Elep_reco', 'theta_reco',
            'reco_numu', 'reco_nc', 'reco_nue',
            'fd_x_vert', 'fd_y_vert', 'fd_z_vert',
        ],
        "noN_sensible" : [
            'nP', 'nipipm', 'nikpm', 'nipi0', 'nik0', 'niem', 'niother',
            'eRecoP', 'eRecoPipm', 'eRecoPi0', 'eRecoOther',
            'Ev_reco', 'Elep_reco', 'theta_reco',
            'muon_tracker', 'muon_contained', 'Ehad_veto',
            'fd_x_vert', 'fd_y_vert', 'fd_z_vert',
        ],
        "noN_sensible2" : [
            'eRecoP', 'eRecoPipm', 'eRecoPi0', 'eRecoOther',
            'Ev_reco', 'Elep_reco', 'theta_reco',
            'muon_tracker', 'muon_contained', 'Ehad_veto',
            'fd_x_vert', 'fd_y_vert', 'fd_z_vert',
        ],
        "noN_sensible3" : [
            'nP', 'nipipm', 'nikpm', 'nipi0', 'nik0', 'niem', 'niother',
            'eRecoP', 'eRecoPipm', 'eRecoPi0', 'eRecoOther',
            'Ev_reco', 'Elep_reco', 'theta_reco',
            'muon_tracker', 'muon_contained', 'Ehad_veto',
            'fd_x_vert_fv_mindist',
            'fd_y_vert_fv_mindist',
            'fd_z_vert_fv_frontdist', 'fd_z_vert_fv_backdist'
        ],
        "noN_trackercontained" : [
            'eRecoP', 'eRecoPip', 'eRecoPim', 'eRecoPi0', 'eRecoOther',
            'Ev_reco', 'Elep_reco', 'theta_reco',
            'muon_tracker', 'muon_contained',
            'reco_numu', 'reco_nc', 'reco_nue', 'reco_lepton_pdg',
            'fd_x_vert', 'fd_y_vert', 'fd_z_vert',
        ],
        "noN_trackercontained_ehadveto" : [
            'eRecoP', 'eRecoPip', 'eRecoPim', 'eRecoPi0', 'eRecoOther',
            'Ev_reco', 'Elep_reco', 'theta_reco',
            'muon_tracker', 'muon_contained', 'Ehad_veto',
            'reco_numu', 'reco_nc', 'reco_nue',
            'fd_x_vert', 'fd_y_vert', 'fd_z_vert',
        ],
        "noN_trackercontained_ehadveto_inputparticles" : [
            'eRecoP', 'eRecoPip', 'eRecoPim', 'eRecoPi0', 'eRecoOther',
            'nP', 'nipip', 'nipim', 'nipi0', 'nipi0', 'nikp', 'nikm', 'nik0', 'niem', 'niother',
            'Ev_reco', 'Elep_reco', 'theta_reco',
            'muon_tracker', 'muon_contained', 'Ehad_veto',
            'reco_numu', 'reco_nc', 'reco_nue',
            'fd_x_vert', 'fd_y_vert', 'fd_z_vert',
        ]
    }
    FAR_RECO_PRESETS = { # (cvn_scores, far_reco)
        "default" : (
            ['fd_numu_score'],
            ['fd_numu_nu_E', 'fd_numu_lep_E', 'fd_numu_had_E']
        ),
        "allcvn" : (
            ['fd_numu_score', 'fd_nue_score', 'fd_nc_score', 'fd_nutau_score'],
            ['fd_numu_nu_E', 'fd_numu_lep_E', 'fd_numu_had_E']
        ),
        "cvn1" : (
            ['fd_numu_score'],
            ['fd_numu_nu_E', 'fd_numu_lep_E', 'fd_numu_had_E']
        ),
        "allcvn_fdnuElast" : (
            ['fd_numu_score', 'fd_nue_score', 'fd_nc_score', 'fd_nutau_score'],
            ['fd_numu_lep_E', 'fd_numu_had_E', 'fd_numu_nu_E']
        ),
        "allcvn_fdlepinfo" : (
            ['fd_numu_score', 'fd_nue_score', 'fd_nc_score', 'fd_nutau_score'],
            ['fd_numu_nu_E', 'fd_numu_lep_E', 'fd_numu_reco_method', 'fd_numu_had_E']
        ),
        "mostcvn" : (
            ['fd_numu_score', 'fd_nue_score', 'fd_nc_score'],
            ['fd_numu_nu_E', 'fd_numu_lep_E', 'fd_numu_had_E']
        ),
        "allcvn_reordered" : (
            ['fd_nue_score', 'fd_nc_score', 'fd_nutau_score', 'fd_numu_score'],
            ['fd_numu_lep_E', 'fd_numu_had_E', 'fd_numu_nu_E']
        ),
        "minimal" : (
            ['fd_numu_score'],
            ['fd_numu_nu_E']
        )
    }
    def __init__(
        self,
        data_path,
        near_reco_preset="noN_sensible3", far_reco_preset="cvn1",
        custom_near_reco=None, custom_far_reco=None,
        sample_weight_var=None, resample_data=None,
        samples_in_val=100_000,
        train=True
    ):

        super().__init__()
        self.data_path = data_path
        self.samples_in_val = samples_in_val
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

        if custom_near_reco is None:
            near_reco = NewPairedData.NEAR_RECO_PRESETS[near_reco_preset]
        else:
            near_reco = custom_near_reco
        if custom_far_reco is None:
            cvn_scores, far_reco = NewPairedData.FAR_RECO_PRESETS[far_reco_preset]
        else:
            cvn_scores, far_reco = custom_far_reco

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

        if self.train:
            data = data[:-self.samples_in_val]
        else:
            data = data[-self.samples_in_val:]

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
