# Facial Expression Recognition (FER-2013) using ResNet50,CNN and VGG16

This repository contains a deep learning pipeline for real-time Facial Expression Recognition (FER).
The project leverages convolutional neural network architectures, specifically a Custom CNN and a ResNet50 model,
to classify human emotions using the FER-2013 dataset. 

A live webcam demonstration script (`test.py`) is included to test the models in real-time.

## Features
* **Automated Data Pipeline:** Code configurations to seamlessly authenticate and download the dataset via the Kaggle API.
* **Dual Model Architectures:** Support for evaluating both a lightweight Custom CNN and a deep ResNet50 architecture.
* **Live Demo:** Real-time facial expression detection using OpenCV and the trained `.keras` models.
* **Accessible Weights:** Pre-trained model weights are hosted on Google Drive for immediate use, bypassing the need to retrain.

##  Dataset
This project uses the `msambare/fer2013` dataset from Kaggle. The dataset consists of grayscale images of faces 
categorized into various emotional expressions (such as sad and surprise).

##  Setup & Installation

### install main libraries 

numpy,pandas,tensorflow,opencv-python,kaggle,scikit-learn and others that are required to run the code and test.py file.

Restnet50.keras download the model file for live demonstration(Recommended):
https://drive.google.com/file/d/18HMKwlVjdNqdH72AiXi-EWaI9_Mzbzoj/view?usp=sharing

Custom_cnn_model.keras download the model file for live demonstration of this model(Recommended):
https://drive.google.com/file/d/1ltv5mD7xsRYNBnPMm78jbGtJtoBsKHPq/view?usp=sharing

If anyone want to run this code then use google colab and enable runtime (T4) cloud base GPU for smooth run.
You can also use kaggle that offer cloud based GPU also

