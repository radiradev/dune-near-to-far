# Near to Far Translation
A generative model based on [min-gpt](https://github.com/karpathy/minGPT). It learns the conditional distribution over the `fd` recontructed variables given `nd` reconstructed variables:
$$p(x_{FD} | x_{ND})$$

We learn this autoregressively, i.e the transformer is trained to predict:
$$p(x_{FD} | x_{ND}) = \prod_i p(x_{i_{FD}}| x_{1_{FD}}, x_{2_{FD}}, ..., x_{3_{FD}}, x_{ND}) $$

## Usage
Change the `data_path` in `NewPairedData` in the `gpt.dataset.py` script.

Train using `python3 gpt_train.py`. 

Then use the `gpt_sample.ipynb` notebook to generate new events and make plots.

## Requirements
Requires an installation of `torch`.

## TO-dos and Issues
[ ] - Generation is not very stable - sometimes creates values outside of the required range for scores and energies.

[ ] - Fix issue with  `best_val_loss` getting overriden at every evaluation in `gpt_train.py`.

[ ] - Explore other generative models. Any conditional generative models should work (diffusion, normalising flows, GANs)


## Variables 
| Far Detector                | CVN scores           | Near Detector        | ND Reco            | Global          |
|-----------------------------|----------------------|----------------------|--------------------|-----------------|
| `fd_numu_nu_E`              | `fd_numu_score`      | `nuPDG`              | `Ev_reco`          | `run`           |
| `fd_numu_had_E`             | `fd_nue_score`       | `nuMomX`             | `Elep_reoo`        | `eventID`       |
| `fd_numu_lep_E`             | `fd_nc_score`        | `NuMomY`             | `theta_reco`       | `isCC`          |
| `fd_numu_recomethod`        | `fd_nutau_score`     | `NuMomZ`             | `eRecoP`           | `nuPDG`         |   
| `numu_longesttrackcontained`| `fd_antinu_score`    | `Ev`                 | `eRecoN`           | `vtxX`          |
| `fd_numu_trackmommethod`    | `fd_proton0_score`   | `mode`               | `eRecoPip`         | `vtxY`          |
| `fd_nue_nu_E`               | `fd_proton1_score`   | `LepPDG`             | `eRecoPim`         | `vtxZ`          |
| `fd_nue_had_E`              | `fd_proton2_score`   | `LepMomX`            | `eRecoPi0`         |                 |
| `fd_nue_lep_E`              | `fd_protonN_score`   | `LepMomY`            | `eRecoOther`       |                 |
| `fd_nue_recomethod`         | `fd_pion0_score`     | `LepMomZ`            |                    |                 |
| `fd_nc_nu_E`                | `fd_pion1_score`     | `lepE`               |                    |                 |
| `fd_nc_had_E`               | `fd_pion2_score`     | `LepNuAngle`         |                    |                 |
| `fd_nc_lep_E`               | `fd_pionN_score`     | `nP`                 |                    |                 |
| `fd_nc_recomethod`          | `fd_pionzero0_score` | `nN`                 |                    |                 |
|                             | `fd_pionzero1_score` | `nPip`               |                    |                 |
|                             | `fd_pionzero2_score` | `nPim`               |                    |                 |
|                             | `fd_pionzeroN_score` | `nPi0`               |                    |                 |
|                             | `fd_neutron0_score`  | `nKp`                |                    |                 |
|                             | `fd_neutron1_score`  | `nKm`                |                    |                 |
|                             | `fd_neutron2_score`  | `nK0`                |                    |                 |
|                             | `fd_neutronN_score`  | `ne`                 |                    |                 |
|                             |                      | `nOther`             |                    |                 |
|                             |                      | `nNucleus`           |                    |                 |
|                             |                      | `nUNKOWN`            |                    |                 |
|                             |                      | `eP`                 |                    |                 |
|                             |                      | `eN`                 |                    |                 |
|                             |                      | `ePip`               |                    |                 |
|                             |                      | `ePim`               |                    |                 |
|                             |                      | `ePi0`               |                    |                 |
|                             |                      | `eOther`             |                    |                 |
|                             |                      | `reco_numu`          |                    |                 |
|                             |                      | `reco_nue`           |                    |                 |
|                             |                      | `reco_nc`            |                    |                 |
|                             |                      | `reco_q`             |                    |                 |
|                             |                      | `muon_contained`     |                    |                 |
|                             |                      | `muon_tracker`       |                    |                 |
|                             |                      | `muon_ecal`          |                    |                 |
|                             |                      | `muon_exit`          |                    |                 |
|                             |                      | `reco_lepton_pdg`    |                    |                 |
|                             |                      | `muon_endpntX`       |                    |                 |
|                             |                      | `muon_endpntY`       |                    |                 |
|                             |                      | `muon_endpntZ`       |                    |                 |
|                             |                      | `Ehad_veto`          |                    |                 |
|                             |                      | `muon_endVolName`    |                    |                 |

### To-do list for larnd-sim translation
- [x] Map just to `fd_nc_nu_E` this contains all hit information - works well for ND Reco, not so well for `larnd-sim`
- [x] Map `larnd-sim` -> `sum(adc)` as a sanity check - seems to be working well, but need a more careful evaluation
- [ ] Map `larndsim` -> `near Ev_reco` seems to not work very well
- [ ] Map ND Reco to `fd_numu_lep_E` or `fd_numu_had_E` to see why mapping to `fd_numu_nu_E` wasn't working.
- [ ] Plot adc sum vs `fc_nc_nu_E`



