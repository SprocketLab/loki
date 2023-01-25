import numpy as np 
import networkx as nx
import json
import pytorch_lightning as pl 
import torch

from torch.utils.data import random_split, DataLoader, TensorDataset
from torch.autograd import Variable
from torchvision import transforms


np.random.seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ImageNetSubset(pl.LightningDataModule):
    def __init__(self, batch_size=64, test_batch_size=512, 
                 tree_structure="mintree", num_of_sampled_classes=1000,
                 basedir="../simclr_imagenet"):
        super().__init__()
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.basedir = basedir
        self.emb_size = 2048
        self.num_classes = 1000

        ## Load tree structure and label info ##
        T = nx.Graph()

        with open(f"{self.basedir}/imagenet_"+tree_structure+".txt", "r") as f:
            for line in f.readlines():
                nodes = line.split()
                for node in nodes:
                    if node not in T:
                        T.add_node(node)
                T.add_edge(*nodes)
                
        leaves = [x for x in T.nodes() if T.degree(x) == 1]
        full_labels_loc = np.array(leaves)
        length = dict(nx.all_pairs_shortest_path_length(T))

        f = open(f"{self.basedir}/dir_label_name.json")
        map_collection = json.load(f)
        f.close()

        ## Compute distance matrix ##
        print("compute distance matrix")
        sampled_classes = np.random.choice(
            len(full_labels_loc), num_of_sampled_classes, replace=False)
        sampled_classes = np.sort(sampled_classes)  

        # NOTE important to implement a self.distance_matrix
        self.distance_matrix = np.zeros(
            (len(full_labels_loc), len(full_labels_loc)))

        for i, each_class_loc_i in enumerate(full_labels_loc):
            for j, each_class_loc_j in enumerate(full_labels_loc):
                distance = length[each_class_loc_i][each_class_loc_j]
                self.distance_matrix[i][j] = distance

        
    def prepare_data(self):
        x_train = np.load(f"{self.basedir}/saved/train_X_1000_classes.npy")
        y_train = np.load(f"{self.basedir}/saved/train_y_1000_classes.npy")
        x_val = np.load(f"{self.basedir}/saved/test_X_1000_classes.npy")
        y_val = np.load(f"{self.basedir}/saved/test_y_1000_classes.npy")

        n_train = y_train.shape[0] # 50_000
        n_val = y_val.shape[0] # 50_000

        idx = np.arange(n_train)
        np.random.shuffle(idx)
        
        # Converting numpy array to Tensor
        self.x_train_tensor = torch.from_numpy(x_train[idx]).float().to(device)
        self.y_train_tensor = torch.from_numpy(y_train[idx]).long().to(device)
        
        self.x_val_tensor = torch.from_numpy(x_val).float().to(device)
        self.y_val_tensor = torch.from_numpy(y_val).long().to(device)
        
        training_dataset = TensorDataset(
            self.x_train_tensor, self.y_train_tensor)
        self.training_dataset = training_dataset
        
        validation_dataset = TensorDataset(
            self.x_val_tensor, self.y_val_tensor)
        self.validation_dataset = validation_dataset

        return training_dataset, validation_dataset
        
    def train_dataloader(self):
        self.prepare_data()
        return DataLoader(self.training_dataset, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return self.test_dataloader() # TODO who cares

    def test_dataloader(self):
        self.prepare_data()
        return DataLoader(self.validation_dataset, 
                          batch_size=self.test_batch_size)


class AnimalKingdom(pl.LightningDataModule):
    def __init__(self, batch_size=64, test_batch_size=512, 
                 tree_structure="animal_kingdom", num_of_sampled_classes=142,
                 basedir="../simclr_imagenet"):
        super().__init__()
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.basedir = basedir
        self.emb_size = 2048
        self.num_classes = 142

        ## Load tree structure and label info ##
        T = nx.Graph()

        with open(f"{self.basedir}/imagenet_"+tree_structure+".txt", "r") as f:
            for line in f.readlines():
                nodes = line.split()
                for node in nodes:
                    if node not in T:
                        T.add_node(node)
                T.add_edge(*nodes)
                
        leaves = [x for x in T.nodes() if T.degree(x) == 1]
        full_labels_loc = np.array(leaves)
        length = dict(nx.all_pairs_shortest_path_length(T))

        f = open(f"{self.basedir}/dir_label_name.json")
        map_collection = json.load(f)
        f.close()

        ## Compute distance matrix ##
        print("compute distance matrix")
        sampled_classes = np.random.choice(
            len(full_labels_loc), num_of_sampled_classes, replace=False)
        sampled_classes = np.sort(sampled_classes)  

        # NOTE important to implement a self.distance_matrix
        self.distance_matrix = np.zeros(
            (len(full_labels_loc), len(full_labels_loc)))

        for i, each_class_loc_i in enumerate(full_labels_loc):
            for j, each_class_loc_j in enumerate(full_labels_loc):
                distance = length[each_class_loc_i][each_class_loc_j]
                self.distance_matrix[i][j] = distance

        
    def prepare_data(self):
        x_train = np.load(f"{self.basedir}/saved/train_X_1000_classes.npy")
        y_train = np.load(f"{self.basedir}/saved/train_y_1000_classes.npy")
        x_val = np.load(f"{self.basedir}/saved/test_X_1000_classes.npy")
        y_val = np.load(f"{self.basedir}/saved/test_y_1000_classes.npy")

        n_train = y_train.shape[0] # 50_000
        n_val = y_val.shape[0] # 50_000

        idx = np.arange(n_train)
        np.random.shuffle(idx)
        
        # Converting numpy array to Tensor
        self.x_train_tensor = torch.from_numpy(x_train[idx]).float().to(device)
        self.y_train_tensor = torch.from_numpy(y_train[idx]).long().to(device)
        
        self.x_val_tensor = torch.from_numpy(x_val).float().to(device)
        self.y_val_tensor = torch.from_numpy(y_val).long().to(device)
        
        training_dataset = TensorDataset(
            self.x_train_tensor, self.y_train_tensor)
        self.training_dataset = training_dataset
        
        validation_dataset = TensorDataset(
            self.x_val_tensor, self.y_val_tensor)
        self.validation_dataset = validation_dataset

        return training_dataset, validation_dataset
        
    def train_dataloader(self):
        self.prepare_data()
        return DataLoader(self.training_dataset, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return self.test_dataloader() # TODO who cares

    def test_dataloader(self):
        self.prepare_data()
        return DataLoader(self.validation_dataset, 
                          batch_size=self.test_batch_size)


class CIFAR100(pl.LightningDataModule):
    def __init__(self, batch_size=64, test_batch_size=512, 
                 tree_structure="parent_child", num_of_sampled_classes=100,
                 basedir="../cifar100"):
        super().__init__()
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.basedir = basedir
        self.emb_size = 2048
        self.num_classes = 100

        ## Load tree structure and label info ##
        T = nx.Graph()

        with open(f"{self.basedir}/cifar_"+tree_structure+".txt", "r") as f:
            for line in f.readlines():
                nodes = line.split()
                for node in nodes:
                    if node not in T:
                        T.add_node(node)
                T.add_edge(*nodes)
                
        leaves = [x for x in T.nodes() if T.degree(x) == 1]
        full_labels_loc = np.array(leaves)
        length = dict(nx.all_pairs_shortest_path_length(T))

        ## Compute distance matrix ##
        print("compute distance matrix")
        sampled_classes = np.random.choice(
            len(full_labels_loc), num_of_sampled_classes, replace=False)
        sampled_classes = np.sort(sampled_classes)  

        # NOTE important to implement a self.distance_matrix
        self.distance_matrix = np.zeros(
            (len(full_labels_loc), len(full_labels_loc)))

        for i, each_class_loc_i in enumerate(full_labels_loc):
            for j, each_class_loc_j in enumerate(full_labels_loc):
                distance = length[each_class_loc_i][each_class_loc_j]
                self.distance_matrix[i][j] = distance
    
    def prepare_data(self):
        x_train = np.load(f"{self.basedir}/cifar_embeddings/cifar100_train_X.npy")
        y_train = np.load(f"{self.basedir}/cifar_embeddings/cifar100_train_y.npy")
        x_val = np.load(f"{self.basedir}/cifar_embeddings/cifar100_test_X.npy")
        y_val = np.load(f"{self.basedir}/cifar_embeddings/cifar100_test_y.npy")

        n_train = y_train.shape[0] # 50_000
        n_val = y_val.shape[0] # 50_000

        idx = np.arange(n_train)
        np.random.shuffle(idx)
        
        # Converting numpy array to Tensor
        self.x_train_tensor = torch.from_numpy(x_train[idx]).float().to(device)
        self.y_train_tensor = torch.from_numpy(y_train[idx]).long().to(device)
        
        self.x_val_tensor = torch.from_numpy(x_val).float().to(device)
        self.y_val_tensor = torch.from_numpy(y_val).long().to(device)
        
        training_dataset = TensorDataset(
            self.x_train_tensor, self.y_train_tensor)
        self.training_dataset = training_dataset
        
        validation_dataset = TensorDataset(
            self.x_val_tensor, self.y_val_tensor)
        self.validation_dataset = validation_dataset

        return training_dataset, validation_dataset

    def train_dataloader(self):
        self.prepare_data()
        return DataLoader(self.training_dataset, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return self.test_dataloader() # TODO who cares

    def test_dataloader(self):
        self.prepare_data()
        return DataLoader(self.validation_dataset, 
                          batch_size=self.test_batch_size)
                          
def main():
    # imagenet = ImageNetSubset()
    # print(imagenet.train_dataloader())

    # cifar100 = CIFAR100()
    # print(cifar100.train_dataloader())

    animal_kingdom = AnimalKingdom()
    print(animal_kingdom.train_dataloader)

if __name__ == "__main__":
    main()
