import torch.nn as nn


def build_model(name: str, num_classes: int, pretrained: bool = False):
    name = name.lower()
    if name in {
        "resnest50",
        "resnest50d",
        "convnext_base",
        "convnext_large",
        "swin_base",
        "swin_large",
        "efficientnetv2_l",
        "vit_b16",
        "vit_l16",
    }:
        try:
            import timm
        except ImportError as exc:
            raise ImportError("timm is required for this model. Install with pip install timm") from exc

        timm_map = {
            "resnest50": "resnest50d",
            "resnest50d": "resnest50d",
            "convnext_base": "convnext_base",
            "convnext_large": "convnext_large",
            "swin_base": "swin_base_patch4_window7_224",
            "swin_large": "swin_large_patch4_window7_224",
            "efficientnetv2_l": "efficientnetv2_l",
            "vit_b16": "vit_base_patch16_224",
            "vit_l16": "vit_large_patch16_224",
        }
        model = timm.create_model(timm_map[name], pretrained=pretrained, num_classes=num_classes)
        return model

    if name in {"inception", "inception_v3"}:
        from torchvision import models

        if pretrained:
            weights = models.Inception_V3_Weights.DEFAULT
        else:
            weights = None
        model = models.inception_v3(weights=weights, aux_logits=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if model.AuxLogits is not None:
            model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        return model

    if name in {"resnet50", "resnet101", "resnet152"}:
        from torchvision import models

        if name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
        elif name == "resnet101":
            weights = models.ResNet101_Weights.DEFAULT if pretrained else None
            model = models.resnet101(weights=weights)
        else:
            weights = models.ResNet152_Weights.DEFAULT if pretrained else None
            model = models.resnet152(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(f"Unknown model: {name}")
