import torch
import pytorch_lightning as pl

import os
from nets.net import LarndSimWrapper
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from MinkowskiEngine.utils import batch_sparse_collate

from pytorch_lightning.loggers import WandbLogger
from larnd_simulation.dataset_larndsim import LarndSimConverted
from torch.utils.data import random_split, DataLoader

# This should be passed from the script that calls this file
dataset = LarndSimConverted()
test_len = int(0.2 * len(dataset))
lengths = [len(dataset) - test_len, test_len]
train_dataset, val_dataset = random_split(dataset, lengths=lengths, generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=batch_sparse_collate, num_workers=128, pin_memory=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=batch_sparse_collate, drop_last=True)


model = LarndSimWrapper()

num_of_gpus = torch.cuda.device_count()
assert num_of_gpus > 0, "This code must be run with at least one GPU"

wandb_logger = WandbLogger(project='3d_to_fd_vars', log_model=all, offline=False)
print(wandb_logger.experiment.dir)
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=str(wandb_logger.experiment.dir),
    monitor='val_loss',
    save_top_k=3,
    mode='min',
)
trainer = pl.Trainer(accelerator="gpu", devices=num_of_gpus, max_epochs=100, strategy='ddp', logger=wandb_logger, callbacks=[checkpoint_callback])
trainer.fit(model, train_loader, val_dataloader)