from larnd_simulation.dataset_larndsim import LarndSimConverted
from nets.net import LarndSimWrapper
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from MinkowskiEngine.utils import batch_sparse_collate

# Adjust if running a new task
target = ['nc_nu_E']
model_task = 'regression'
ckpt_path = '/global/homes/r/rradev/near_to_far/wandb/latest-run/files/epoch=21-step=4642.ckpt'

# load the data
dataset = LarndSimConverted(targets=target)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=batch_sparse_collate, pin_memory=True, num_workers=12)
# evaluate 


model = LarndSimWrapper(model_task=model_task, num_outputs=len(target))
# load the model
trainer = pl.Trainer(accelerator='gpu', limit_test_batches=0.01)

trainer.predict(model, dataloader, ckpt_path=ckpt_path)


