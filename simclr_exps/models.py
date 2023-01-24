import pytorch_lightning as pl
import torch.nn as nn
import torch

from torchmetrics import Accuracy
from metrics import MeanSquaredDistance
from torch import optim
from loki import loki_ste_predict, loki_polytope_predict

class SingleLayerModel(pl.LightningModule):
    def __init__(self, emb_size, k, dists):
        super().__init__()
        self.emb_size = emb_size
        self.k = k
        self.dists = torch.Tensor(dists).cuda()

        self.model = nn.Sequential(nn.Linear(emb_size, k))
        self.loss = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        preds = self.model(x)
        loss = self.loss(preds, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        preds = self.model(x)
        loss = nn.functional.cross_entropy(preds, y)
        accuracy = Accuracy(task="multiclass", num_classes=self.k).cuda()
        meansquareddist = MeanSquaredDistance().cuda()
        acc = accuracy(preds, y)
        msd = meansquareddist(preds.argmax(dim=1), y, self.dists)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.log("val_dist", msd, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        preds = self.model(x)
        loss = nn.functional.cross_entropy(preds, y)
        accuracy = Accuracy(task="multiclass", num_classes=self.k).cuda()
        meansquareddist = MeanSquaredDistance().cuda()
        acc = accuracy(preds, y)
        msd = meansquareddist(preds.argmax(dim=1), y, self.dists)
        self.log("test_acc", acc, on_epoch=True)
        self.log("test_dist", msd, on_epoch=True)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    

class SingleLayerLokiModel(pl.LightningModule):
    def __init__(self, emb_size, k, dists, model=None, ste=False):
        super().__init__()
        self.emb_size = emb_size
        self.k = k
        self.ste = ste

        if model is None:
            self.model = nn.Sequential(nn.Linear(emb_size, k))
        else: 
            self.model = model

        self.dists = torch.Tensor(dists).cuda()
        self.loss_loki = nn.NLLLoss()
        self.loss_clf = nn.CrossEntropyLoss()

    def forward_loki(self, batch):
        x, y = batch
        x = x.view(x.size(0), -1)
        logits = self.model(x)
        # Loki
        probs = logits.softmax(dim=1)
        if self.ste:
            preds = loki_ste_predict(probs, self.dists)
        else:
            preds = loki_polytope_predict(probs, self.dists)
        eps = 0.0001
        smooth_onehot = torch.ones(self.k, self.k) * (1 / (self.k - 1)) * eps
        smooth_onehot += torch.eye(self.k) * ((1 - eps) \
                                               - (1 / (self.k - 1)) * eps)
        smooth_onehot = smooth_onehot.cuda()
        preds_smoothed = preds @ smooth_onehot
        #loss_loki = self.loss_loki( # self training
        #    torch.log(preds_smoothed), logits.argmax(dim=1))
        loss_loki = self.loss_loki(torch.log(preds_smoothed), y)
        #loss = self.loss_clf(logits, y) + loss_loki
        loss = loss_loki
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.forward_loki(batch)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, preds, y = self.forward_loki(batch)
        accuracy = Accuracy(task="multiclass", num_classes=self.k).cuda()
        meansquareddist = MeanSquaredDistance().cuda()
        acc = accuracy(preds, y)
        msd = meansquareddist(preds.argmax(dim=1), y, self.dists)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.log("val_dist", msd, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss)
    
    def test_step(self, batch, batch_idx):
        loss, preds, y = self.forward_loki(batch)
        accuracy = Accuracy(task="multiclass", num_classes=self.k).cuda()
        meansquareddist = MeanSquaredDistance().cuda()
        acc = accuracy(preds, y)
        msd = meansquareddist(preds.argmax(dim=1), y, self.dists)
        self.log("test_acc", acc, on_epoch=True)
        self.log("val_dist", msd, on_epoch=True)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    

def main():
    emb_size = 2048
    model = SingleLayerModel(emb_size=emb_size, k=1000)
    print(model)

if __name__ == "__main__":
    main()
