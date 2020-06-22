import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os

class DataLoaders:

    def __init__(self, data_path):
        self.data_path = data_path

    def image_transformations(self):

        return {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }


    def build_data_loaders(self):

        data_transforms = self.image_transformations()

        image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_path, x),
                                                  data_transforms[x])
                          for x in ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                      shuffle=True, num_workers=4)
                       for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        return dataloaders, dataset_sizes