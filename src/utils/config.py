from omegaconf import OmegaConf
from pydantic import BaseModel


class Config(BaseModel):
    batch_size: int
    data_dir: str
    name_model: str
    test_size: float
    n_classes: int
    num_epochs: int
    step_size: int
    seed: int
    gamma: float
    beta: float
    threshold: float
    pretrained: bool
    device: str
    criterion: str
    encoder: str
    encoder_kwargs: dict
    criterion_kwargs: dict
    optimizer: str
    optimizer_kwargs: dict
    scheduler: str
    scheduler_kwargs: dict

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
