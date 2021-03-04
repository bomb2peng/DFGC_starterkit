from PIL import Image
from torch.utils import data
import torch


## ---------------------- Dataloaders ---------------------- ##
class Dataset_Csv(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, folders, labels, transform=None):
        "Initialization"
        # self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, use_transform):
        image = Image.open(path)
        if use_transform is not None:
            image = use_transform(image)
        return image

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        X = self.read_images(folder, self.transform)  # (input) spatial images
        y = torch.LongTensor([self.labels[index]])  # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return X, y


## ---------------------- end of Dataloaders ---------------------- ##



