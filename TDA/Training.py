import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from model import TransferModel
from dataclass import tf_test, tf_train, Target
import random

# Set seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seeds()

class LightningTransferModel(pl.LightningModule):
    def __init__(self, num_class, base_net='resnet152', use_bottleneck=True, bottleneck_width=256, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = TransferModel(num_class, base_net, use_bottleneck, bottleneck_width)

        state_dict = torch.load('/home/amirkh/Python/Main/adapted-model/final_model_6.pt')
        self.model.load_state_dict(state_dict, strict=True)

        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_class)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_class)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_class)

        

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        acc = self.train_acc(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        acc = self.val_acc(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss, 'val_acc': acc}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        acc = self.test_acc(y_hat, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'test_loss': loss, 'test_acc': acc}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def plot_confusion_matrix(self, cm, class_names):
        plt.figure(figsize=(15, 15))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix\nAccuracy: {self.trainer.callback_metrics["test_acc"]:.2f}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        self.logger.experiment.add_figure('confusion_matrix', plt.gcf(), self.current_epoch)
        plt.close()


# Main execution
if __name__ == "__main__":
    # Define paths and parameters
    data_dir = '/home/amirkh/Python/data/Detect dataset/Cropped images'
    csv_file = '/home/amirkh/Python/Main/CSV/Fin1-3(6).csv'
    

    # Set up datasets
    train_dataset = Target(csv_file, data_dir, split='train', split_column='0', transform=tf_train, dataset_type='F1')
    val_dataset = Target(csv_file, data_dir, split='val', split_column='0', transform=tf_test, dataset_type='F1')
    test_dataset = Target(csv_file, data_dir, split='test', split_column='0', transform=tf_test, dataset_type='F1')

    # Set up data loaders
    batch_size = 64
    num_workers = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    # Set up model
    model = LightningTransferModel(num_class=6, base_net='resnet152', use_bottleneck=True, bottleneck_width=256, learning_rate=0.1)

    # Load pre-trained weights


    # Set up trainer
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max', save_top_k=1)
    logger = TensorBoardLogger("lightning_logs", name="transfer_model")
    trainer = pl.Trainer(max_epochs=5, logger=logger, callbacks=[checkpoint_callback])

    # Train and test
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)

    print("Training, validation, and testing completed.")