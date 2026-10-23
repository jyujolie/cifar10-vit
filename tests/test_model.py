import torch

from src.model import build_model


def test_vit_produces_ten_logits_per_image():
    model = build_model(pretrained=False)
    model.eval()
    with torch.inference_mode():
        logits = model(torch.zeros(2, 3, 224, 224))

    assert logits.shape == (2, 10)
