

# MLOps Zoomcamp Final Project 2025

This repository contains the final project for the [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)

[![Run Unit Tests](https://github.com/facug91/mlops-zoomcamp-final-project/actions/workflows/python-tests.yml/badge.svg)](https://github.com/facug91/mlops-zoomcamp-final-project/actions/workflows/python-tests.yml)

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

## Building Docker Images

This project provides two Docker environments:

- **CPU version** located in `docker/cpu`
- **GPU version** located in `docker/gpu`

Build Arguments

- `ENV=prod`: production environment (default packages only)
- `ENV=dev`: development environment (includes dev dependencies)

Both versions include:

- `requirements.txt` always installed
- `requirements-dev.txt` installed **only** when `ENV=dev`

**Note:**
If you only want to run the service and use the pre-built images, you can pull them directly from my GitHub Container Registry without building locally:
- ghcr.io/facug91/ml-dev-env-cpu:latest
- ghcr.io/facug91/ml-prod-env-cpu:latest
- ghcr.io/facug91/ml-dev-env-gpu:latest
- ghcr.io/facug91/ml-prod-env-gpu:latest

### Included Tools and Libraries

Both CPU and GPU environments come pre-installed with:

- **Experiment Tracking & Model Registry**
  - [MLflow](https://mlflow.org/)

- **Machine Learning framework**
  - [PyTorch](https://pytorch.org/) and [TorchVision](https://pytorch.org/vision/stable/index.html)

- **Cloud and Storage**
  - [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
  - [botocore](https://botocore.amazonaws.com/)

- **Databases**
  - [psycopg2-binary](https://pypi.org/project/psycopg2-binary/)

Development images (`ENV=dev`) also include extra dependencies for debugging and development.

- **Workflow Orchestration**
  - [Prefect](https://www.prefect.io/)

- **Unit and integration testing**
  - [pytest](https://docs.pytest.org/)

- **Machine Learning tools**
  - [scikit-learn](https://scikit-learn.org/stable/)
  - [tqdm](https://tqdm.github.io/)

### **Building the CPU image**

```bash
docker build \
  -f docker/cpu/Dockerfile \
  --build-arg ENV=dev \
  -t ml-dev-env-cpu \
  ./docker/cpu
```

### **Building the GPU image**

Requirements:

- [NVIDIA driver 570 or above (compatible with CUDA 12.8)](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html)

- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

```bash
docker build \
  -f docker/gpu/Dockerfile \
  --build-arg ENV=dev \
  -t ml-dev-env-gpu \
  ./docker/gpu
```

## Running the Development or Production Environment

This project uses [Docker Compose](https://docs.docker.com/compose/) to manage all services, with support for **profiles** to choose which environment to start.

### Profiles Available

The `docker-compose.yml` defines multiple profiles so you can choose what environment to run:

- `prod-cpu` → Production environment (CPU version)
- `prod-gpu` → Production environment (GPU version, requires NVIDIA GPU)
- `dev-cpu` → Development environment (CPU version) (includes production environment)
- `dev-gpu` → Development environment (GPU version, requires NVIDIA GPU) (includes production environment)

### Starting the environment

**Example: Development with GPU**

```bash
docker compose --profile dev-gpu up --build
```

_Replace the profile name according to your needs._

### Working inside the container

When running a dev-* profile, you can attach VS Code to the development container using the Dev Containers extension:
1. In VS Code, open the command palette (Ctrl+Shift+P).
2. Select `Dev Containers: Attach to Running Container...`
3. Choose the container:
  - For GPU dev: ml-dev-env-gpu
  - For CPU dev: ml-dev-env-cpu

Once attached, you can run scripts inside the container, from the /workspace directory.
