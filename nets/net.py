import torch
from models.voxel_convnext import VoxelConvNeXtClassifier
import pytorch_lightning as pl
from MinkowskiEngine import SparseTensor
import numpy as np
import torchmetrics
import torch.nn.functional as F


class LarndSimWrapper(pl.LightningModule):
    def __init__(self, model_task, num_outputs):
        super().__init__()
        self.model_task = model_task
        self.model = VoxelConvNeXtClassifier(in_chans=1, D=3, num_classes=num_outputs, drop_path_rate=0.0)
        self.loss = torch.nn.MSELoss()

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
        if self.model_task == 'classification':
            labels = torch.argmax(labels, dim=1)
    
        elif self.model_task == 'regression':
            labels = torch.log(torch.clamp(labels, 0.03)) # don't have to change labels

        elif self.model_task == 'probability_regression':
            predictions = self.model(stensor)
            predictions = F.log_softmax(predictions, dim=1)
            labels = torch.log(labels)

        loss = self.loss(predictions, labels)
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
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        loss, predictions, labels = self._shared_step(batch, batch_idx)
        return {'predictions': predictions, 'labels': labels, 'loss': loss}
    
    def predict_epoch_end(self, outputs):
        all_predictions = torch.cat([x['predictions'] for x in outputs])
        all_labels = torch.cat([x['labels'] for x in outputs])
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        return {'predictions': all_predictions, 'labels': all_labels, 'avg_loss': avg_loss}


    def test_step(self, batch, batch_idx):
        loss, predictions, labels = self._shared_step(batch, batch_idx)
        batch_size = labels.shape[0]
        self.log('test_loss', loss, batch_size=batch_size, prog_bar=True)
        return {'test_loss': loss, 'predictions': predictions, 'labels': labels}


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-6)
        return optimizer

