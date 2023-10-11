import torch
import timm
from src.utils.config import Config


class LandscapeClassifier(torch.nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self._model = timm.create_model(config.name_model, pretrained=config.pretrained)
        num_features = self._model.fc.in_features
        self._model.fc = torch.nn.Linear(num_features, config.n_classes)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._model(tensor)
