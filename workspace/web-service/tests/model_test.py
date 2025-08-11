import pytest
import torch
from services.model import ModelService
from PIL import Image
from unittest.mock import patch, MagicMock


@pytest.fixture
def dummy_image():
    """
    Create a 100x100 red image for testing purposes.
    This avoids the need for external image files.
    """
    return Image.new("RGB", (100, 100), color="red")


@pytest.fixture
def service():
    """
    Fixture: Returns a ModelService instance without loading a model.
    This avoids slow I/O and GPU usage during tests.
    """
    return ModelService(model_name=None)


def test_preprocess_shape_and_range(service, dummy_image):
    """
    Test that preprocess returns a normalized tensor of the expected shape.
    Also ensures values are within a reasonable normalized range.
    """
    tensor = service.preprocess(dummy_image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
    assert torch.all(tensor <= 3) and torch.all(tensor >= -3)


def test_postprocess_sorted_and_labels(service):
    """
    Test that postprocess:
    - Returns a list of (label, probability) tuples.
    - Is sorted by probability descending.
    """
    dummy_output = torch.tensor([0.1, 0.5, 0.2, 0.1, 0.1])
    results = service.postprocess(dummy_output)
    probs = [p for _, p in results]
    assert probs == sorted(probs, reverse=True)
    labels = [label for label, _ in results]
    assert set(labels) == set(service.class_index.values())


def test_process_applies_softmax(service):
    """
    Test that process applies softmax to the model's raw outputs.
    """
    mock_model = MagicMock(return_value=torch.tensor([[1.0, 2.0, 3.0, 0.5, 0.1]]))
    service.model = mock_model
    input_tensor = torch.randn(1, 3, 224, 224)
    probs = service.process(input_tensor)
    assert torch.isclose(torch.sum(probs), torch.tensor(1.0), atol=1e-5)


def test_predict_calls_all_methods_in_order(service, dummy_image):
    """
    Test that predict calls preprocess, process and postprocess and returns the result of postprocess.
    """
    with patch.object(service, "preprocess", return_value="tensor") as mock_pre:
        with patch.object(service, "process", return_value="processed") as mock_proc:
            with patch.object(service, "postprocess", return_value="final") as mock_post:
                result = service.predict(dummy_image)
                assert result == "final"
                mock_pre.assert_called_once_with(dummy_image)
                mock_proc.assert_called_once_with("tensor")
                mock_post.assert_called_once_with("processed")
