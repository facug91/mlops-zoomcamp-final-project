import pytest
import torch
from services.model import ModelService
from PIL import Image
from unittest.mock import patch


@pytest.fixture
def dummy_image():
    return Image.new("RGB", (1280, 720), color="red")


@pytest.fixture
def service():
    return ModelService(model_name=None)


def test_preprocess_shape(service, dummy_image):
    tensor = service.preprocess(dummy_image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)


def test_postprocess_output_format(service):
    dummy_output = torch.tensor([0.1, 0.5, 0.2, 0.1, 0.1])
    results = service.postprocess(dummy_output)
    assert isinstance(results, list)
    assert all(isinstance(label, str) and isinstance(prob, float) for label, prob in results)
    assert abs(sum(prob for _, prob in results) - 1.0) < 1e-3


def test_predict_returns_sorted(service, dummy_image):
    with patch.object(service, "process", return_value=torch.tensor([0.1, 0.4, 0.3, 0.1, 0.1])):
        results = service.predict(dummy_image)
        probs = [prob for _, prob in results]
        assert probs == sorted(probs, reverse=True)

