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
from gpt.dataset import NewPairedData
from gpt.model import GPT
from gpt.trainer import Trainer
from gpt.utils import set_seed, setup_logging, CfgNode as CN
from helpers import (
    read_reweight_dir, get_reweight_uniform, get_reweight_scalefactors, get_resample_data
)

PRINT_MODEL=True

# -----------------------------------------------------------------------------

def get_config(work_dir):
    C = CN()

    # system
    C.system = CN()
    C.system.seed = None
    C.system.work_dir = work_dir

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'
    C.model.no_causal_near_mask = False

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster

    # dataset
    C.dataset = CN()
    C.dataset.near_reco_preset="noN_sensible3"
    C.dataset.far_reco_preset="cvn1"
    C.dataset.samples_in_val=300_000

    return C

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@torch.no_grad()
def estimate_loss(val_loader, sample_weighting):
    model.eval()
    losses = []
    for i, batch in enumerate(val_loader):
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
        "--resampling_ndcaf_Ev",
        action="store_true",
        help="Sample data at load time with the overall Ev from the ND CAFS"
    )
    g.add_argument(
        "--resampling_osc_Ev",
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

    resample_data = get_resample_data(args)

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
            data_path=args.data_path,
            near_reco_preset=config.dataset.near_reco_preset,
            far_reco_preset=config.dataset.far_reco_preset,
            sample_weight_var=sample_weight_var,
            samples_in_val=config.dataset.samples_in_val,
            train=True
        )
        val_dataset = NewPairedData(
            data_path=args.data_path,
            near_reco_preset=config.dataset.near_reco_preset,
            far_reco_preset=config.dataset.far_reco_preset,
            sample_weight_var=sample_weight_var,
            samples_in_val=config.dataset.samples_in_val,
            train=False
        )
        config.model.block_size = train_dataset.get_block_size()
        config.model.near_reco_size = train_dataset.get_near_reco_length()
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
            data_path=args.data_path,
            near_reco_preset=config.dataset.near_reco_preset,
            far_reco_preset=config.dataset.far_reco_preset,
            resample_data=resample_data,
            samples_in_val=config.dataset.samples_in_val,
            train=True
        )
        val_dataset = NewPairedData(
            data_path=args.data_path,
            near_reco_preset=config.dataset.near_reco_preset,
            far_reco_preset=config.dataset.far_reco_preset,
            samples_in_val=config.dataset.samples_in_val,
            train=False
        )

        config.model.block_size = train_dataset.get_block_size()
        config.model.near_reco_size = train_dataset.get_near_reco_length()
        config.model.far_reco_size = train_dataset.get_far_reco_length()
        config.model.scores_size = train_dataset.get_scores_length()

        model = GPT(config.model)

        trainer = Trainer(config.trainer, model, train_dataset)

    val_loader = DataLoader(
        val_dataset, shuffle=False, pin_memory=True, batch_size=512, num_workers=4
    )

    if PRINT_MODEL:
        model.eval()
        idx = torch.tensor(
            val_dataset.data[:, :len(val_dataset.near_reco)], dtype=torch.float
        ).to(device)[0:1]
        model.print_forward_pass(idx)
        model.train()

    i = 0
    while os.path.exists(os.path.join(config.system.work_dir, f"losses_{i}.txt")):
        i += 1
    loss_file = os.path.join(config.system.work_dir, f"losses_{i}.txt")
    print(f"Loss file is {loss_file}")
    best_val_loss = torch.inf
    n_plateau = 0
    manual_plateau_scheduler = False
    # iteration callback
    def batch_end_callback(trainer):
        global best_val_loss
        global n_plateau
        global loss_file

        if isinstance(trainer.lr_scheduler, torch.optim.lr_scheduler.OneCycleLR):
            trainer.lr_scheduler.step()

        if trainer.iter_num % 100 == 0:
            acc_loss = float(np.mean(trainer.running_losses))
            trainer.running_losses.clear()
            print(
                f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: "
                f"train loss {acc_loss:.5f}"
            )
            with open(loss_file, "a+") as f:
                f.write(f"TRAIN {trainer.iter_num} {acc_loss:.6f}\n")

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                val_loss = estimate_loss(val_loader, reweighting)
                print("Validation Loss:", val_loss)

            with open(loss_file, "a+") as f:
                f.write(f"VALID {trainer.iter_num} {val_loss:.6f}\n")

            # save the latest model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("Model has the best validation loss, saving model")
                ckpt_path = os.path.join(config.system.work_dir, "model.pt")
                torch.save(model.state_dict(), ckpt_path)
                n_plateau = 0
            elif manual_plateau_scheduler:
                n_plateau += 1
                if n_plateau > 3:
                    for g in trainer.optimizer.param_groups:
                        print(f"LR: {g['lr']} -> {g['lr'] * 0.5}")
                        with open(loss_file, "a+") as f:
                            f.write(f"LR {trainer.iter_num} {g['lr']}\n")
                        g["lr"] = g["lr"] * 0.5
                        n_plateau = 0

            if not manual_plateau_scheduler:
                for g in trainer.optimizer.param_groups:
                    print(f"LR: {g['lr']}")
                    with open(loss_file, "a+") as f:
                        f.write(f"LR {trainer.iter_num} {g['lr']}\n")
                    
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()

    print(f"best val loss: {best_val_loss}")

