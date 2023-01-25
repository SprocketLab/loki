import fire
import torch
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from datasets import ImageNetSubset
from models import SingleLayerModel, SingleLayerLokiModel

def main(data="imagenet", epochs=10, 
         batch_size=128, test_batch_size=512, ste=False, negiden=False,
         pretraining=True):
    
    print(f"USING STE={ste}")
    print(f"USING NEGIDEN={negiden}")
    
    torch.set_float32_matmul_precision('medium')
    wandb_logger = WandbLogger(name='Nick', project='Loki')

    if data == "imagenet":
        dataset = ImageNetSubset(
            batch_size=batch_size, 
            test_batch_size=test_batch_size)
    else:
        raise NotImplementedError
    
    model = SingleLayerModel(emb_size=dataset.emb_size, 
                             dists=dataset.distance_matrix,
                             k=dataset.num_classes)
    
    train_loader = dataset.train_dataloader()
    valid_loader = dataset.val_dataloader()
    test_loader = dataset.test_dataloader()

    if pretraining:
        print("⚡" * 20 + " TRAINING " + "⚡" * 20)
        trainer = pl.Trainer(max_epochs=100, 
                            enable_checkpointing=False, 
                            accelerator="gpu", logger=wandb_logger)
        trainer.fit(model, train_loader, valid_loader)
        trainer.test(model, test_loader)

    print("⚡" * 20 + " FINE TUNING " + "⚡" * 20)
    model_ft = SingleLayerLokiModel(emb_size=dataset.emb_size, 
                                 k=dataset.num_classes,  
                                 dists=dataset.distance_matrix,
                                 model=model.model,
                                 ste=ste, negiden=negiden)
    trainer_ft = pl.Trainer(max_epochs=100, 
                         enable_checkpointing=False, 
                         accelerator="gpu", logger=wandb_logger)
    trainer_ft.fit(model_ft, train_loader, valid_loader)
    trainer_ft.test(model_ft, test_loader)


if __name__ == "__main__":
    fire.Fire(main)
