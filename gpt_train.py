"""
Train a model on a 2D gaussian
"""

import os
import sys
import torch
import argparse
import glob
from torch.utils.data.dataloader import DataLoader

import numpy as np
from gpt.dataset import PairedData, NewPairedData
from gpt.model import GPT
from gpt.trainer import Trainer
from gpt.utils import set_seed, setup_logging, CfgNode as CN

# -----------------------------------------------------------------------------

def get_config(work_dir):

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = work_dir

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster

    return C

eval_iters = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@torch.no_grad()
def estimate_loss(val_loader, sample_weighting):
    model.eval()
    losses = []
    for batch in (val_loader):
        batch = [t.to(device) for t in batch]
        if sample_weighting:
            x, y, weight_var = batch
            logits, loss = model(x, y, sample_weights_var=weight_var)
        else:
            x, y = batch
            logits, loss = model(x, y)
        losses.append(loss.item())
    loss = np.stack(losses).mean()
    model.train()
    return loss

def read_reweight_dir(reweight_dir):
    bins_file = glob.glob(os.path.join(reweight_dir, "*_bins.npy"))
    assert len(bins_file) == 1, "Invalid reweight dir structure."
    weight_bins = np.load(bins_file[0])
    hist_file = glob.glob(os.path.join(reweight_dir, "*_hist.npy"))
    assert len(hist_file) == 1, "Invalid reweight dir structure."
    weight_hist = np.load(hist_file[0])
    var_file = glob.glob(os.path.join(reweight_dir, "*_var.txt"))
    assert len(var_file) == 1, "Invalid reweight dir structure."
    with open(var_file[0], "r") as f:
        var_name = f.read().rstrip("\n")
    return weight_bins, weight_hist, var_name

# Reweights s.t. the most energies are flat and the rest is almost flat
# (very large weights # at the extreme energies can make training unstable)
def get_reweight_uniform(train_sample_weight_var_data):
    bins = np.arange(0.0, 14.25, 0.25)
    train_hist, _ = np.histogram(train_sample_weight_var_data, bins=bins)
    train_hist = train_hist.astype(float)
    train_hist /= np.sum(train_hist)
    target_hist = np.ones_like(train_hist).astype(float)
    target_hist /= np.sum(target_hist)
    ratio_hist = target_hist / train_hist

    bins = np.concatenate([bins, [120.0]])
    ratio_hist = np.concatenate([ratio_hist, [np.max(ratio_hist)]])
    ratio_hist = np.clip(ratio_hist, 0.0, 20.0)

    print("Training sample weights histogram is:")
    print(ratio_hist)
    print(bins)

    return ratio_hist, bins

def get_reweight_scalefactors(train_sample_weight_var_data, target_bins, target_hist):
    train_hist, train_bins = np.histogram(train_sample_weight_var_data, bins=target_bins)
    train_hist = train_hist.astype(float)
    # Fairly sure this is the wrong thing to do... the normalisation of each histogram before
    # taking the ratio should be 1 / sum(counts) not 1 / sum(rates).
    # for i in range(len(train_hist)):
    #     train_hist[i] /= (train_bins[i + 1] - train_bins[i])
    train_hist /= np.sum(train_hist)
    ratio_hist = target_hist / train_hist

    # Dont really care about <0.5GeV and >6GeV
    ratio_hist[-2:] = 1.0
    ratio_hist[0] = 1.0

    print("Training sample weights histogram is:")
    print(ratio_hist)
    print(train_bins)

    return ratio_hist, train_bins

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("data_path", type=str, help="Training data csv.")
    parser.add_argument("work_dir", type=str, help="Experiment directory, put these in out/.")

    parser.add_argument(
        "-o", "--config_override",
        type=str, action="append", default=None,
        help=(
            "Argument override for the CfgNode, "
            "string should be like 'arg=value' e.g. 'model.n_gaussians=30'. "
            "Can repeat this argument."
        )
    )

    g = parser.add_mutually_exclusive_group()
    g.add_argument(
        "--training_reweight",
        type=str, default=None,
        help=(
            "Weight training samples to a flux. "
            "A dir containing three files: thebin edges (*_bins.npy), bin count (*_hist.npy), "
            "and weighting variable name (*_var.txt)."
        )
    )
    g.add_argument(
        "--uniform_reweight_Ev",
        action="store_true",
        help="Reweight such that in most energies the number of events is uniform in Ev"
    )
    g.add_argument(
        "--uniform_reweight_fd_numu_nu_E",
        action="store_true",
        help="Reweight such that in most energies the number of events is uniform in fd_numu_nu_E"
    )
    g.add_argument(
        "--uniform_resampling_Ev",
        action="store_true",
        help="Sample data at load time with a flat Ev"
    )
    g.add_argument(
        "--uniform_resampling_fd_numu_nu_E",
        action="store_true",
        help="Sample data at load time with a flat fd_numu_nu_E"
    )
    g.add_argument(
        "--uniform_resampling_ndcaf_Ev",
        action="store_true",
        help="Sample data at load time with the overall Ev from the ND CAFS"
    )
    g.add_argument(
        "--uniform_resampling_osc_Ev",
        action="store_true",
        help="Sample data at load time with the oscillated Ev (target of the PRISM LC)"
    )

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_arguments()

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    # get default config and overrides from the command line, if any
    config = get_config(args.work_dir)
    if args.config_override is not None:
        overrides = [ "--" + override for override in args.config_override ]
        config.merge_from_args(overrides)
    setup_logging(config)
    set_seed(config.system.seed)
    print(config)

    reweighting = (
        args.training_reweight is not None or
        args.uniform_reweight_Ev or
        args.uniform_reweight_fd_numu_nu_E
    )
    
    if args.uniform_resampling_Ev:
        uniform_resample_data = (np.array([0.5, 6.0]), np.array([0.5]), "Ev")
    elif args.uniform_resampling_fd_numu_nu_E:
        uniform_resample_data = (np.array([0.5, 6.0]), np.array([0.5]), "fd_numu_nu_E")
    elif args.uniform_resampling_ndcaf_Ev:
        bins = np.load("data/ndcafs_all_oa_trueE/allCAF_Ev_oaall_bins.npy")
        hist = np.load("data/ndcafs_all_oa_trueE/allCAF_Ev_oaall_hist.npy") # expect bin counts not rate
        hist = hist[(bins >= 0.5) & (bins <= 6.0)]
        bins = bins[(bins >= 0.5) & (bins <= 6.0)]
        hist /= np.sum(hist)
        uniform_resample_data = (bins, hist, "Ev")
    elif args.uniform_resampling_osc_Ev:
        bins = np.load("data/prism_nufit_target_fd_flux_norate/FDTargetFlux_bins.npy")
        hist = np.load("data/prism_nufit_target_fd_flux_norate/FDTargetFlux_hist.npy") # expect bin counts not rate
        hist = hist[(bins >= 0.5) & (bins <= 6.0)]
        bins = bins[(bins >= 0.5) & (bins <= 6.0)]
        hist /= np.sum(hist)
        uniform_resample_data = (bins, hist, "Ev")
    else:
        uniform_resample_data = None

    if reweighting:
        print(f"Reweighting training using {args.training_reweight}")

        if args.uniform_reweight_Ev:
            sample_weight_var = "Ev"
        elif args.uniform_reweight_fd_numu_nu_E:
            sample_weight_var = "fd_numu_nu_E"
        else:
            weights_bins, weights_hist, sample_weight_var = read_reweight_dir(
                args.training_reweight
            )

        train_dataset = NewPairedData(
            data_path=args.data_path, train=True, sample_weight_var=sample_weight_var
        )
        val_dataset = NewPairedData(
            data_path=args.data_path, train=False, sample_weight_var=sample_weight_var
        )
        config.model.block_size = train_dataset.get_block_size()
        config.model.far_reco_size = train_dataset.get_far_reco_length()
        config.model.scores_size = train_dataset.get_scores_length()

        if args.uniform_reweight_Ev or args.uniform_reweight_fd_numu_nu_E:
            weights_hist, weights_bins = get_reweight_uniform(train_dataset.data[:, -1])
        else:
            weights_hist, weights_bins = get_reweight_scalefactors(
                train_dataset.data[:, -1], weights_bins, weights_hist
            )
        np.save(os.path.join(args.work_dir, "sampling_weights_hist.npy"), weights_hist)
        np.save(os.path.join(args.work_dir, "sampling_weights_bins.npy"), weights_bins)
        with open(os.path.join(args.work_dir, "sampling_weights_var.txt"), "w") as f:
            f.write(sample_weight_var + "\n")

        model = GPT(config.model, sample_weights_data=(weights_hist, weights_bins))

        trainer = Trainer(config.trainer, model, train_dataset, sample_weighting=True)

    else:
        train_dataset = NewPairedData(
            data_path=args.data_path, train=True, uniform_resample=uniform_resample_data
        )
        val_dataset = NewPairedData(data_path=args.data_path, train=False)

        config.model.block_size = train_dataset.get_block_size()
        config.model.far_reco_size = train_dataset.get_far_reco_length()
        config.model.scores_size = train_dataset.get_scores_length()

        model = GPT(config.model)

        trainer = Trainer(config.trainer, model, train_dataset)

    val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            pin_memory=True,
            batch_size=512,
            num_workers=4
    )

    best_val_loss = torch.inf
    # iteration callback
    def batch_end_callback(trainer):
        global best_val_loss

        if trainer.iter_num % 10 == 0:
            print(
                f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: "
                f"train loss {trainer.loss.item():.5f}"
            )

        if trainer.iter_num % 300 == 0:
            # evaluate both the train and test score
            print(
                f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: "
                f"train loss {trainer.loss.item():.5f}"
            )
            model.eval()
            with torch.no_grad():
                val_loss = estimate_loss(val_loader, reweighting)
                print("Validation Loss:", val_loss)

            # save the latest model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("Model has the best validation loss, saving model")
                ckpt_path = os.path.join(config.system.work_dir, "model.pt")
                torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()

