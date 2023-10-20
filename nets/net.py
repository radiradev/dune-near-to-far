import torch
from models.voxel_convnext import VoxelConvNeXtClassifier
import pytorch_lightning as pl
from MinkowskiEngine import SparseTensor
import numpy as np
import torchmetrics
import torch.nn.functional as F


class LarndSimWrapper(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = VoxelConvNeXtClassifier(in_chans=1, D=3, num_classes=4, drop_path_rate=0.0)
        self.loss = torch.nn.L1Loss()
    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, batch_idx):
        coordinates, energy, labels = batch
        stensor = SparseTensor(
            features=energy,
            coordinates=coordinates,
            device=self.device
        )

        labels = labels.squeeze()
        predictions = self.model(stensor)
        predictions = F.log_softmax(predictions, dim=1)
        log_labels = torch.log(labels)
        loss = self.loss(predictions, log_labels)
        return loss, predictions, labels
    
    def training_step(self, batch, batch_idx):
        if self.global_step % 50 == 0:
            torch.cuda.empty_cache()
        loss, _, _ = self._shared_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, predictions, labels = self._shared_step(batch, batch_idx)
        batch_size = labels.shape[0]
        self.log('val_loss', loss, batch_size=batch_size, sync_dist=True)
        return loss


    def test_step(self, batch, batch_idx):
        loss, predictions, labels = self._shared_step(batch, batch_idx)
        batch_size = labels.shape[0]
        self.log('test_loss', loss, batch_size=batch_size, prog_bar=True)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

