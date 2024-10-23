Retraining VGG for Custom Image Classification using Transfer Learning

This repository contains code for performing transfer learning with the VGG-16/19 model. We leverage a pretrained VGG model, originally trained on ImageNet, and retrain it for a new image classification task. This experiment demonstrates how to reuse the feature extraction power of VGG for custom datasets and achieve high accuracy with minimal training.

Key Features:

Pretrained VGG-16/19 model used for feature extraction.
Transfer learning by adding custom fully connected layers for specific classification tasks.
Training and validation plots to monitor model performance over time.
Code designed for use in Azure Machine Learning Studio, integrating cloud-based training.
Watch the Full Tutorial:

Check out the video tutorial on YouTube for a detailed, step-by-step guide on how to implement this code in Azure ML Studio and retrain the VGG model for your own dataset:

Watch the YouTube Tutorial on Retraining VGG in Azure ML

Getting Started:

Clone this repo to your local machine or Azure ML workspace.
Run the Jupyter notebook to start training the VGG model with your own dataset.
Requirements:

Python 3.x
TensorFlow/Keras
Azure ML Studio
