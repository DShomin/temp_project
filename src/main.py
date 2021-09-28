import pandas as pd
import numpy as np
import os

from torch.utils.data import DataLoader
import random
import torch
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint#, StochasticWeightAveraging
# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from warmup_scheduler import GradualWarmupScheduler
from typing import Optional

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold

from model import ImgModel
from data import WavDatset
from trans import get_transforms

BASE_DIR = '../data/'
TRAIN_PATH = os.path.join(BASE_DIR, 'train_dataset')
TEST_PATH = os.path.join(BASE_DIR, 'test_dataset')

class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class LitClassifier(pl.LightningModule):
    """
    >>> LitClassifier(Backbone())  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    LitClassifier(
      (backbone): ...
    )
    """

    def __init__(
        self,
        scale_list = [0.25, 0.5], # 0.125, 
        backbone: Optional[ImgModel] = None,
        learning_rate: float = 0.0001,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone'])
        if backbone is None:
            backbone = ImgModel()
        self.backbone = backbone
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        out = self.backbone.forward_features(batch)
        out = self.pool(out)
        out = self.fc(out[:,:,0,0])
        return out

    def training_step(self, batch, batch_idx):
        
        x, y = batch
        labels = y.long()

        output = self.backbone(x)

        loss = self.criterion(output, labels)
        
        try:
            pred = output.detach().cpu()
            pred = torch.argmax(pred, dim=1)
            acc=accuracy_score(labels.detach().cpu(), pred) 

            self.log("acc", acc, on_step= True, prog_bar=True, logger=True)
            self.log("Train Loss", loss, on_step= True,prog_bar=True, logger=True)
        
        except:
            pass

        return {"loss": loss, "predictions": output.detach().cpu(), "labels": labels.detach().cpu()}

    def training_epoch_end(self, outputs):

        preds = []
        labels = []
        
        for output in outputs:
            
            preds += output['predictions']
            labels += output['labels']

        labels = torch.stack(labels)
        preds = torch.stack(preds)
        

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.backbone(x)
        labels = y.long()

        loss = self.criterion(output, labels)
        
        self.log('val_loss', loss, on_step= True, prog_bar=True, logger=True)
        return {"predictions": output, "labels": labels}

    def validation_epoch_end(self, outputs):

        preds = []
        labels = []
        for output in outputs:
            # preds.append(output['predictions'])
            # labels.append(output['labels'])
            preds += output['predictions']
            labels += output['labels']
        labels = torch.stack(labels)
        preds = torch.stack(preds)
        preds = preds.detach().cpu()
        preds = torch.argmax(preds, dim=1)
        val_acc=accuracy_score(labels.detach().cpu(), preds)
        self.log("val_acc", val_acc, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        out = self.backbone(batch)
        return out

    def configure_optimizers(self):

        param_optimizer = list(self.backbone.named_parameters()) # self.model.named_parameters()
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 1e-6,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_parameters, lr=self.hparams.learning_rate)
        scheduler_cosie = CosineAnnealingLR(optimizer, T_max= 10, eta_min=1e-6, last_epoch=-1)
        # scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler_cosie)
        return dict(optimizer=optimizer, lr_scheduler=scheduler_cosie) # , lr_scheduler=scheduler_warmup lr_scheduler=scheduler[optimizer], [scheduler]

class MyDataModule(pl.LightningDataModule):

    def __init__(
        self,
        TRAIN_DF,
        VALID_DF,
        batch_size: int = 32,
    ):
        super().__init__()
        
        trn_dataset = WavDatset(TRAIN_DF, trans=get_transforms(data='train')) 
        val_dataset = WavDatset(VALID_DF, trans=get_transforms(data='valid')) 
        
        self.train_dset = trn_dataset
        self.valid_dset = val_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dset, batch_size=self.batch_size, shuffle=True, num_workers=6)

    def val_dataloader(self):
        return DataLoader(self.valid_dset, batch_size=self.batch_size, shuffle=False, num_workers=6) 

def cli_main():
    logger = WandbLogger(name=f'CNN_eff0_fold{C_FOLD}', project='sp_recog')
    classifier =  LitClassifier()
    mc = ModelCheckpoint('model', monitor='val_acc', mode='max', filename='{epoch}-{val_acc:.4f}_' + f'fold_{C_FOLD}')
    # swa = StochasticWeightAveraging(swa_epoch_start=2, annealing_epochs=2)
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=1,
        # accelerator='ddp_spawn',
        # stochastic_weight_avg=True,
        callbacks=[mc],
        logger=logger
        )
    mydatamodule = MyDataModule(TRAIN_DF, VALID_DF)
    trainer.fit(classifier, datamodule=mydatamodule)

if __name__ == '__main__':
    sample_idx = 1000
    train_df = pd.read_csv('../data/train_label.csv')#[:sample_idx]

    # split fold
    C_FOLD = 0
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    for i, (train_idx, valid_idx) in enumerate(skf.split(X=train_df, y=train_df['age_'])):

        if i == C_FOLD:
            TRAIN_DF = train_df.iloc[train_idx]

            VALID_DF = train_df.iloc[valid_idx]

    seed_everything()
    cli_main()

