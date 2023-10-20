from larnd_simulation.dataset_larndsim import LarndSimConverted
from nets.net import LarnSimWrapper
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from MinkowskiEngine.utils import batch_sparse_collate

dataset = LarndSimConverted()
dataloader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=batch_sparse_collate)
# evaluate 

model = LarnSimWrapper()
# load the model
trainer = pl.Trainer(accelerator='gpu')
ckpt_path = None
trainer.test(model, dataloader, ckpt_path=ckpt_path)

