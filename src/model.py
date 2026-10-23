"""Create the ViT Tiny model used by this project."""

import timm

MODEL_NAME = "vit_tiny_patch16_224.augreg_in21k_ft_in1k"


def build_model(*, pretrained: bool = True):
    """Return a ViT Tiny with a new 10-class classification head."""
    return timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=10)
