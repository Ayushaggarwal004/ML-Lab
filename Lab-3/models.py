# pyright: reportMissingImports=false
import torch
from torch import nn


class MLP(nn.Module):
    """Flatten 28x28 images and classify them with the specified MLP."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


class CNN(nn.Module):
    """The CNN architecture specified in Lab 3."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(32 * 7 * 7, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(images)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        return self.classifier(x)


class PatchAttention(nn.Module):
    """16 non-overlapping 7x7 patches and a two-layer Transformer encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.patch_projection = nn.Linear(7 * 7, 64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(64, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        patches = images.unfold(2, 7, 7).unfold(3, 7, 7)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(batch_size, 16, 49)
        tokens = self.patch_projection(patches)
        encoded_tokens = self.transformer(tokens)
        pooled_tokens = encoded_tokens.mean(dim=1)
        return self.classifier(pooled_tokens)


def build_models() -> dict[str, nn.Module]:
    return {
        "MLP": MLP(),
        "CNN": CNN(),
        "Attention": PatchAttention(),
    }
