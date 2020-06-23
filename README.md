# Covid-19 x-ray image classification

- Dataset: `https://github.com/ieee8023/covid-chestxray-dataset`
- Model: Pre-trained resnet18

To run:
1. Download full dataset to `artifacts/dataset` in this project repo, with the following paths populated:
    
   - `artifacts/dataset/annotations`
   - `artifacts/dataset/images`
   - `artifacts/dataset/metadata.csv`

2. Make sure you have docker installed & set current working directory to `covid-xray-classifier` project.
3. Build docker image in terminal: `make Build` 
4. Run training: `make Train`

Some useful code borrowed from: `https://github.com/bentrevett/pytorch-image-classification/blob/master/5%20-%20ResNet.ipynb`