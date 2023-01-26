import fire
import torch
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from datasets import ImageNetSubset, CIFAR100, AnimalKingdom
from models import SingleLayerModel, SingleLayerLokiModel
from models import MultiLayerModel, MultiLayerLokiModel

def main(data="imagenet",
         batch_size=128, test_batch_size=512, ste=False, negiden=False,
         pretraining=True):
    
    #print(f"USING STE={ste}")
    #print(f"USING NEGIDEN={negiden}")
    
    torch.set_float32_matmul_precision('medium')
    wandb_logger = WandbLogger(name='Nick', project='Loki')

    if data == "imagenet":
        dataset = ImageNetSubset(
            batch_size=batch_size, 
            test_batch_size=test_batch_size)
        ModelPre = SingleLayerModel
        ModelFt = SingleLayerLokiModel
        epochs = 100
        epochs_ft = 100
        print(f"USING DATASET={data}")
    elif data == "animalkingdom":
        dataset = AnimalKingdom(
            batch_size=batch_size, 
            test_batch_size=test_batch_size)
        ModelPre = SingleLayerModel
        ModelFt = SingleLayerLokiModel
        epochs = 20
        epochs_ft = 100
        print(f"USING DATASET={data}")
    elif data == "cifar100":
        dataset = CIFAR100(
            batch_size=batch_size, 
            test_batch_size=test_batch_size)
        ModelPre = SingleLayerModel
        ModelFt = SingleLayerLokiModel
        epochs = 10
        epochs_ft = 100
        print(f"USING DATASET={data}")
    else:
        raise NotImplementedError
    
    model = ModelPre(emb_size=dataset.emb_size, 
                             dists=dataset.distance_matrix,
                             k=dataset.num_classes)
    
    train_loader = dataset.train_dataloader()
    valid_loader = dataset.val_dataloader()
    test_loader = dataset.test_dataloader()

    if pretraining:
        print("⚡" * 20 + " TRAINING " + "⚡" * 20)
        trainer = pl.Trainer(max_epochs=epochs, 
                            enable_checkpointing=False, 
                            accelerator="gpu", logger=wandb_logger)
        trainer.fit(model, train_loader, valid_loader)

        print("EVALUATING UNDER BASELINE")
        trainer.test(model, test_loader)

    print("⚡" * 20 + " FINE TUNING " + "⚡" * 20)
    model_ft = ModelFt(emb_size=dataset.emb_size, 
                                 k=dataset.num_classes,  
                                 dists=dataset.distance_matrix,
                                 model=model.model,
                                 ste=ste, negiden=negiden)
    trainer_ft = pl.Trainer(max_epochs=epochs_ft, 
                         enable_checkpointing=False, 
                         accelerator="gpu", logger=wandb_logger)
    
    print("EVALUATING UNDER LOKI")
    trainer_ft.test(model_ft, test_loader)

    trainer_ft.fit(model_ft, train_loader, valid_loader)
    trainer_ft.test(model_ft, test_loader)


if __name__ == "__main__":
    fire.Fire(main)
