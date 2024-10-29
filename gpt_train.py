"""
Train a model on a 2D gaussian
"""

import os
import sys
import torch
import argparse
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
            print(val_loss, best_val_loss)

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
