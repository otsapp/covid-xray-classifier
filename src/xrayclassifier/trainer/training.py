import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
from datetime import date
import os

from xrayclassifier.processing import PrepareImageDirectory
from xrayclassifier.processing import DataLoaders
from xrayclassifier.trainer.process import ModelTraining


def model_fn(model_path, name):
    model = TheModelClass(*args, **kwargs)
    model_file_path = os.path.join(model_path, name)
    model.load_state_dict(torch.load(model_file_path))
    return model.eval()


def main(data_path, images_path, train_path, val_path, train_ratio, num_epochs, model_path, name):

    print("> Preparing directories")
    PrepareImageDirectory(train_ratio, data_path, images_path, train_path, val_path).build_train_val_directory()

    print("> Building torch dataloaders")
    data_loaders, dataset_sizes = DataLoaders(data_path).build_data_loaders()

    print("> Downloading model and adding new layers")
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features

    # adding extra layers
    model.fc = nn.Sequential(nn.Linear(num_ftrs, 512),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(512, 10),
                             nn.LogSoftmax(dim=1))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    print("> Training model")
    model = ModelTraining(data_loaders, dataset_sizes, model, criterion, optimizer, scheduler, num_epochs).train_model()
    print("> Training complete")

    # save model state_dict
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    model_file_path = os.path.join(model_path, name)

    torch.save(model, model_file_path)
    print(f"> Model saved at: {model_file_path}")





if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-path', help='Path for data', default='artifacts/dataset')
    parser.add_argument('-i', '--images-path', help='Path for images', default='split_images')
    parser.add_argument('-t', '--train-path', help='Path for model', default='train')
    parser.add_argument('-v', '--val-path', help='Path for model', default='val')
    parser.add_argument('-r', '--train-ratio', help='Percentage of images to be trained on', default=0.8)
    parser.add_argument('-e', '--num-epochs', help='Number of epochs', default=8)
    parser.add_argument('-m', '--model-path', help='path to models', default='artifacts/models')
    parser.add_argument('-n', '--name', help='model file name', default=str(date.today()))
    args = vars(parser.parse_args())

    main(**args)
