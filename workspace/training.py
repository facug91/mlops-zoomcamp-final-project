import argparse
import json
import os
import time

import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V3_Small_Weights, ResNet50_Weights, mobilenet_v3_small, resnet50
from tqdm import tqdm

import mlflow
from prefect import flow, task

MLFLOW_URL = os.getenv("MLFLOW_URL", "http://mlflow:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "fruits-classifier")


def configure_mlflow():
    print("MLFlow remote server: ", MLFLOW_URL)
    print("MLFlow experiment name: ", MLFLOW_EXPERIMENT)
    mlflow.set_tracking_uri(MLFLOW_URL)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


@task(log_prints=True)
def get_data_loaders(data_dir: str):

    print("Creating data loaders with dataset ", data_dir)

    mlflow.log_param("dataset_path", data_dir)

    input_size = 224
    batch_size = 64

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        ),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "val", "test"]
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == "train"), num_workers=4)
        for x in ["train", "val", "test"]
    }

    class_names = image_datasets["train"].classes
    num_classes = len(class_names)

    print("Data loaders created successfully")

    return (dataloaders, class_names, num_classes)


@task(log_prints=True)
def initialize_model(model_name, num_classes):
    mlflow.log_param("model_name", model_name)

    if model_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        mlflow.log_param("model_pretrained_weights", str(weights))

        for param in model.parameters():
            param.requires_grad = False

        # Replace final FC layer
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )
    elif model_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = mobilenet_v3_small(weights=weights)
        mlflow.log_param("model_pretrained_weights", str(weights))

        for param in model.parameters():
            param.requires_grad = False

        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )
        for param in model.classifier[3].parameters():
            param.requires_grad = True

    else:
        raise ValueError(f"Model {model_name} is not supported. Choose 'resnet50' or 'mobilenet_v3_small'.")

    return model


@task(log_prints=True)
def train_model(model, model_name, device, dataloaders, num_epochs=10):

    print("Training begin")

    criterion = nn.CrossEntropyLoss()
    if model_name == "resnet50":
        optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    elif model_name == "mobilenet_v3_small":
        optimizer = optim.Adam(model.classifier[3].parameters(), lr=1e-3)
    else:
        raise ValueError(f"Model {model_name} is not supported. Choose 'resnet50' or 'mobilenet_v3_small'.")

    mlflow.log_param("optimizer", optimizer.__class__.__name__)
    mlflow.log_param("lr", optimizer.param_groups[0]["lr"])
    mlflow.log_param("epochs", num_epochs)
    mlflow.log_param("loss_function", criterion.__class__.__name__)

    best_acc = -1.0

    model = model.to(device)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Wrap dataloader with tqdm for progress bar
            loop = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Phase", leave=False)

            for inputs, labels in loop:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Update tqdm description dynamically
                loop.set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                        "Batch Acc": f"{(preds == labels).float().mean().item():.4f}",
                    }
                )

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            mlflow.log_metric(f"{phase}_loss", epoch_loss, step=epoch)
            mlflow.log_metric(f"{phase}_acc", epoch_acc, step=epoch)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Save the best model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model, "models/best-model.pth")

    print(f"\nBest Validation Accuracy: {best_acc:.4f}")

    model = torch.load("models/best-model.pth", weights_only=False)
    os.remove("models/best-model.pth")
    torch.save(model.state_dict(), f"models/{model_name}_weights.pth")
    mlflow.pytorch.log_model(model, artifact_path="models", registered_model_name=f"fruits-classifier-{model_name}")

    return model


@task(log_prints=True)
def eval(model, device, dataloaders, class_names):

    print("Evaluating best model with test data.")

    model.eval()
    all_preds = []
    all_labels = []
    acc_execution_time = 0.0

    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:

            start_time = time.time()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs).to("cpu")
            _, preds = torch.max(outputs, 1)
            end_time = time.time()
            execution_time = end_time - start_time

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            acc_execution_time += execution_time

    avg_execution_time_ms = acc_execution_time * 1000.0 / len(dataloaders["test"].dataset)
    avg_execution_time_ms = int(avg_execution_time_ms * 100.0) / 100.0
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    for label, metrics in report.items():
        label = label.replace(" ", "_").replace("-", "_")
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                metric_name = metric_name.replace(" ", "_").replace("-", "_")
                mlflow.log_metric(f"{label}_{metric_name}", value)
        else:
            mlflow.log_metric(label, metrics)
    mlflow.log_metric("avg_execution_time_ms", avg_execution_time_ms)

    with open("metrics/classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
    mlflow.log_artifact("metrics/classification_report.json", artifact_path="models")

    with open("models/class_names.txt", "w") as f:
        f.write("\n".join(class_names))
    mlflow.log_artifact("models/class_names.txt", artifact_path="models")

    print(report)


@flow(log_prints=True)
def training_flow(model_name="mobilenet_v3_small"):

    print("Executing training flow")

    configure_mlflow()

    data_dir = "./data/fruits_dataset"

    with mlflow.start_run():

        mlflow.set_tag("Data Scientist", "Facundo Galán")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device :", device)

        mlflow.log_param("device", device)

        dataloaders, class_names, num_classes = get_data_loaders(data_dir)

        model = initialize_model(model_name, num_classes)

        model = train_model(model, model_name, device, dataloaders, num_epochs=10)

        eval(model, device, dataloaders, class_names)

    print("Training flow completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for fruit classification using MLflow and Prefect.")
    parser.add_argument(
        "--model-name",
        choices=["mobilenet_v3_small", "resnet50"],
        required=True,
        help="Name of the model to be trained.",
    )

    args = parser.parse_args()
    training_flow(args.model_name)
