"""Small synthetic examples and a purpose-built demonstration head."""
import torch
from torch import nn
from torch.nn import functional as F
from core.sapam import SA_PAM


class DemoSegmenter(nn.Module):
    """A small demonstration model, not a manuscript backbone."""
    def __init__(self, c: dict, use_prior: bool):
        super().__init__()
        self.use_prior = use_prior
        if use_prior:
            self.features = SA_PAM(embed_dim=c["embed_dim"], num_heads=c["heads"],
                                   window_size=c["window"], num_scales=2)
        else:
            self.features = nn.Sequential(nn.Conv2d(3, c["embed_dim"], 3, padding=1), nn.ReLU())
        self.head = nn.Conv2d(c["embed_dim"], c["classes"], 1)

    def forward(self, rgb: torch.Tensor, prior=None) -> torch.Tensor:
        if self.use_prior and prior is None:
            raise ValueError("The fine-stage demonstration requires a prior")
        features = self.features(rgb, prior, 1) if self.use_prior else self.features(rgb)
        return self.head(features)


def synthetic_scene(c: dict, seed: int) -> tuple:
    generator = torch.Generator().manual_seed(seed)
    size = c["image_size"]
    rgb = torch.rand(2, 3, size, size, generator=generator)
    labels = (rgb[:, 0] + rgb[:, 1] > 1).long()
    coarse_rgb = F.avg_pool2d(rgb, 2)
    coarse_labels = F.interpolate(labels[:, None].float(), scale_factor=0.5,
                                  mode="nearest")[:, 0].long()
    return rgb, labels, coarse_rgb, coarse_labels
