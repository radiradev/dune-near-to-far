"""
Train a model on a 2D gaussian
"""

import os
import sys
import torch
from torch.utils.data.dataloader import DataLoader

import numpy as np
from gpt.dataset import PairedData
from gpt.model import GPT
from gpt.trainer import Trainer
from gpt.utils import set_seed, setup_logging, CfgNode as CN

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/gpt_nd2fd'

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


if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    train_dataset = PairedData()
    val_dataset = PairedData(train=False)
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
    model = GPT(config.model)

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)
    
    global best_val_loss
    best_val_loss = 1000 
    # iteration callback
    def batch_end_callback(trainer):
        global best_val_loss
        best_val_loss = 1000 
        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 250 == 0:
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