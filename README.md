# Covid-19 x-ray image classification

#### Output: 
Builds an image classifier using transfer learning that detects xray images of 'covid-19' vs. 'other' patient's lungs.

#### Overview: 
Model is extremely basic, doesn't use localisation and validation accuracy can vary between 75% to 87% based on several runs.

- Dataset: `https://github.com/ieee8023/covid-chestxray-dataset`
- Model: resnet18 pre-trained on ImageNet.
- Project modularised and set up to be data agnostic so they be adapted to build other classifiers.
- src/xrayclassifier/trainer/training.py contains main run process

#### Local model training:
1. Download full dataset to `artifacts/dataset` in this project repo, with the following paths populated:
    
   - `artifacts/dataset/annotations`
   - `artifacts/dataset/images`
   - `artifacts/dataset/metadata.csv`

2. Make sure you have docker installed & set current working directory to `covid-xray-classifier` project.
3. Build docker image in terminal: `make Build` 
4. Run training: `make Train`

#### Possible to dos:
- Prediction module.
- Add Jenkinsfile and marathon.json for pipeline and app definition.
- Add data source as container volume to avoid bringing data into project files directly.
- Use bounding box annotations supplied with xray images for more accurate classification.
- Use more advanced models such as image segmentation.

Some useful code borrowed from: `https://github.com/bentrevett/pytorch-image-classification/blob/master/5%20-%20ResNet.ipynb`