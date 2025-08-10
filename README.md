

# MLOps Zoomcamp Final Project 2025

This repository contains the final project for the [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)

## Project: Fruits Classification

![Fruits example](./images/fruit_banner.png)

### Problem description

Accurate classification of fruits from images is an important task in applications such as automated retail systems, agriculture monitoring, and inventory management.  
In real-world scenarios, fruit sorting and identification are often done manually, which can lead to human errors, inefficiencies, and increased labor costs.  
Automating this process can improve speed, accuracy, and scalability in environments such as:  
- Smart supermarkets — automatic detection of fruits at checkout.
- Food supply chains — quality control and classification in packaging facilities.
- Robotics — enabling fruit-handling robots to recognize and sort products.

This project implements a deep learning-based fruit classifier that takes an image as input and outputs the predicted fruit category along with the confidence scores for all classes.  
The model is deployed as a REST API, enabling integration with other services and workflows.  

### Dataset description

The dataset used in this project is the public [Fruits Classification Dataset](https://www.kaggle.com/datasets/utkarshsaxenadn/fruits-classification/data) from Kaggle, containing images of 5 fruit categories:  
- Apple
- Banana
- Grape
- Mango
- Strawberry

Each image is RGB and belongs to one of the five classes, stored in separate directories for training, validation, and test sets.
The dataset is already preprocessed and balanced, making it suitable for supervised classification tasks without additional labeling.

### Repository scope and objective

This repository contains the complete workflow for training, deploying, and serving a fruit image classification model, including:
- Model training and evaluation using PyTorch and a fine-tuned ResNet-50 architecture.
- Model serving through a Flask-based API with a /predict endpoint that accepts uploaded images.
- MLOps integrations:
  - Logging predictions and execution metrics into a PostgreSQL database.
  - Storing uploaded images in MinIO (S3-compatible) and linking them with database records.
  - Experiment tracking using MLflow.
- Infrastructure and automation:
  - Docker-based containerization for reproducible environments.
  - GitHub Actions for CI pipeline execution and testing.
  - Modular architecture separating model logic, metrics storage, and API service.
