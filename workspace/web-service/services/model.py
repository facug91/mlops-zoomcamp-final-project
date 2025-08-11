from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


class ModelService:
    """
    Service class for loading a fine-tuned ResNet50 model,
    preprocessing images, performing inference, and returning predictions.
    """

    def __init__(self, model_name: str = "mobilenet_v3_small", class_index: dict = None):
        """
        Initialize the model and image transform pipeline.

        Args:
            model_path (str): Path to the saved model weights.
            class_index (dict): Mapping from class index to label name.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_size = 224
        self.transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.class_index = class_index or {0: "Apple", 1: "Banana", 2: "Grape", 3: "Mango", 4: "Strawberry"}
        if model_name is not None:  # Useful for unit tests
            self.model_name = model_name
            self.model_path = f"/root/workspace/models/{model_name}_weights.pth"
            self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        """
        Load the fine-tuned ResNet50 model from disk.

        Returns:
            torch.nn.Module: The model ready for inference.
        """
        if self.model_name == "mobilenet_v3_small":
            model = models.mobilenet_v3_small()
            in_features = model.classifier[3].in_features
            model.classifier[3] = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, len(self.class_index)),
            )
        else:
            model = models.resnet50()
            model.fc = nn.Sequential(
                nn.Linear(model.fc.in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, len(self.class_index)),
            )
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess a PIL image into a normalized tensor.

        Args:
            image (PIL.Image.Image): Input image.

        Returns:
            torch.Tensor: Normalized 4D tensor ready for inference.
        """
        image = image.convert("RGB")
        return self.transform(image).unsqueeze(0).to(self.device)

    def process(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform inference on the input tensor.

        Args:
            input_tensor (torch.Tensor): Preprocessed image tensor.

        Returns:
            torch.Tensor: Output probabilities tensor.
        """
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        return probabilities

    def postprocess(self, output_tensor: torch.Tensor) -> List[Tuple[str, float]]:
        """
        Convert model output tensor into sorted class labels with probabilities.

        Args:
            output_tensor (torch.Tensor): Output from the model.

        Returns:
            List[Tuple[str, float]]: Sorted (label, probability) pairs.
        """
        probabilities = output_tensor.cpu().numpy()
        indices = np.argsort(probabilities)[::-1]
        probabilities = probabilities.tolist()
        return [(self.class_index[idx], int(probabilities[idx] * 1000) / 1000.0) for idx in indices]

    def predict(self, image: Image.Image) -> List[Tuple[str, float]]:
        """
        Predict the top classes for a given image.

        Args:
            image (PIL.Image.Image): Image to classify.

        Returns:
            List[Tuple[str, float]]: Sorted (label, probability) pairs.
        """
        input_tensor = self.preprocess(image)
        output_tensor = self.process(input_tensor)
        result = self.postprocess(output_tensor)
        return result


# Example usage
if __name__ == "__main__":
    service = ModelService()
    image = Image.open("/root/workspace/data/fruits_dataset/test/Banana/Banana (3813).jpeg").convert("RGB")
    predictions = service.predict(image)
    for label, prob in predictions:
        print(f"{label}: {prob:.4f}")
