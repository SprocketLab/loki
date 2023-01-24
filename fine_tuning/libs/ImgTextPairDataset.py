from torch.utils.data import Dataset
from PIL import Image

class ImgTextPairDataset(Dataset):
    def __init__(self, dataset, preprocess):
        self.dataset = dataset
        self.preprocess = preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, targets = self.dataset[idx]
        image = self.preprocess(image).unsqueeze(0)
        return image, targets