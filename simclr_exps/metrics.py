import torch
from torchmetrics import Metric
    

class MeanSquaredDistance(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("ds", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, 
               dists: torch.Tensor):
        assert preds.shape == target.shape
        self.ds += (dists[preds, target] ** 2).sum()
        self.total += target.numel()

    def compute(self):
        return self.ds.float() / self.total
    