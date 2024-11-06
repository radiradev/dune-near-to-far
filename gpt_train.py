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
def estimate_loss(val_loader):
    model.eval()
    losses = []
    for batch in (val_loader):
        batch = [t.to(device) for t in batch]
        x, y = batch
        logits, loss = model(x, y)
        losses.append(loss.item())
    loss = np.stack(losses).mean()
    model.train()
    return loss

def get_reweight_scalefactors(train_fd_nu_E, reweight_dir):
    bins_file = glob.glob(os.path.join(reweight_dir, "*_bins.npy"))
    assert len(bins_file) == 1, "Invalid rewight dir structure."
    bins_file = bins_file[0]
    target_bins = np.load(bins_file)
    hist_file = glob.glob(os.path.join(reweight_dir, "*_hist.npy"))
    assert len(hist_file) == 1, "Invalid rewight dir structure."
    hist_file = hist_file[0]
    target_hist = np.load(hist_file)

    train_hist, train_bins = np.histogram(train_fd_nu_E, bins=target_bins)
    train_hist = train_hist.astype(float)
    for i in range(len(train_hist)):
        train_hist[i] /= (train_bins[i + 1] - train_bins[i])
    train_hist /= np.sum(train_hist)
    ratio_hist = target_hist / train_hist

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

    parser.add_argument(
        "--training_reweight",
        type=str, default=None,
        help=(
            "Weight training samples to a flux."
            "A dir containing two files for the bin edges (*_bins.npy) and count (*_hist.npy)."
        )
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

    train_dataset = NewPairedData(data_path=args.data_path, train=True)
    val_dataset = NewPairedData(data_path=args.data_path, train=False)
    val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            pin_memory=True,
            batch_size=512,
            num_workers=4
        )
    # construct the model
    # config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    config.model.far_reco_size = train_dataset.get_far_reco_length()
    config.model.scores_size = train_dataset.get_scores_length()
    
    if args.training_reweight is not None:
        print(f"Reweighting training using {args.training_reweight}")
        fd_numu_nu_E_input_col_idx = (
            len(train_dataset.near_reco) +
            len(train_dataset.cvn_scores) +
            train_dataset.far_reco.index("fd_numu_nu_E")
        )
        weights_hist, weights_bins = get_reweight_scalefactors(
            train_dataset.data[:, fd_numu_nu_E_input_col_idx], args.training_reweight
        )
        np.save(os.path.join(args.work_dir, "sampling_weights_hist.npy"), weights_hist)
        np.save(os.path.join(args.work_dir, "sampling_weights_bins.npy"), weights_bins)
        model = GPT(
            config.model,
            sample_weights_data=(
                weights_hist,
                weights_bins,
                fd_numu_nu_E_input_col_idx - len(train_dataset.near_reco)
            )
        )
    else:
        model = GPT(config.model)

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)
    
    best_val_loss = torch.inf
    # iteration callback
    def batch_end_callback(trainer):
        global best_val_loss

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 300 == 0:
            # evaluate both the train and test score

            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
            model.eval()
            with torch.no_grad():   
                val_loss = estimate_loss(val_loader)
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
