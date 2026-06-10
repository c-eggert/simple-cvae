"""
Evaluation script for the MNIST CVAE.

Before running, train the model with cvae_train.py and point CHECKPOINT_PATH
at the saved file you want to evaluate.
"""

import copy
from pathlib import Path
from typing import Tuple

import torch
from torch import nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST

import cvae
from cvae import (
    ConditionSchema,
    ConditionVariable,
    InputEncoderCategoricalToOneHot,
    MSEByCategoryEvaluation,
    ConditionalPredictionEvaluation,
)
from cvae.cvae import EncoderBase, DecoderBase, ConvBlock, UpConvBlock


CHECKPOINT_PATH = "checkpoints/epoch_0200_loss0.0031426929035313.pth"
OUTPUT_DIR = "eval_results"
LATENT_DIM = 16


class Encoder(EncoderBase):
    def __init__(self, in_channels_data: int, in_channels_cond: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvBlock(in_channels_data, in_channels_cond, 16,  4, 1, padding=1)
        self.conv2 = ConvBlock(16,               in_channels_cond, 32,  8, 2, padding=1)
        self.conv3 = ConvBlock(32,               in_channels_cond, 64,  8, 2, padding=1)
        self.conv4 = ConvBlock(64,               in_channels_cond, 128, 8, 1, padding=0)
        self.latent_mu     = nn.Linear(128, out_channels)
        self.latent_logvar = nn.Linear(128, out_channels)

    def forward(self, data: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(data, condition)
        x = self.conv2(x, condition)
        x = self.conv3(x, condition)
        x = self.conv4(x, condition)
        x = torch.squeeze(x, dim=[-2, -1])
        return self.latent_mu(x), self.latent_logvar(x)


class Decoder(DecoderBase):
    def __init__(self, in_latent_dim: int, in_channels_cond: int, out_channels: int):
        super().__init__(in_latent_dim)
        self.unproject = nn.Linear(in_latent_dim, 128)
        self.up1  = UpConvBlock(128, in_channels_cond, 64, 8, 1, padding=1)
        self.up2  = UpConvBlock(64,  in_channels_cond, 32, 8, 1, padding=1)
        self.up3  = UpConvBlock(32,  in_channels_cond, 16, 8, 1, padding=0)
        self.up4  = UpConvBlock(16,  in_channels_cond, 8,  4, 1, padding=1)
        self.pred = nn.Conv2d(8, out_channels, kernel_size=1)

    def forward(self, sampled: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = self.unproject(sampled)
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        x = self.up1(x, condition)
        x = self.up2(x, condition)
        x = self.up3(x, condition)
        x = self.up4(x, condition)
        return self.pred(x)


condition_schema = ConditionSchema([
    ConditionVariable("digit", InputEncoderCategoricalToOneHot(num_classes=10)),
])

COND_DIM = condition_schema.output_dim  # 10


transform_image = transforms.Compose([
    transforms.Resize((11, 11), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.float()),
])
transform_label = transforms.Lambda(lambda y: condition_schema.encode({"digit": y}))

mnist_test = MNIST(
    root="data", train=False, download=False,
    transform=transform_image, target_transform=transform_label,
)


encoder = Encoder(in_channels_data=1, in_channels_cond=COND_DIM, out_channels=LATENT_DIM)
decoder = Decoder(in_latent_dim=LATENT_DIM, in_channels_cond=COND_DIM, out_channels=1)
model = cvae.CVAE(encoder=encoder, decoder=decoder)

checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


mse_eval = MSEByCategoryEvaluation(model=model, schema=condition_schema)
mse_eval.run(dataset=mnist_test, output_dir=OUTPUT_DIR)


cond_pred_eval = ConditionalPredictionEvaluation(
    model=model,
    schema=condition_schema,
    predictor_encoder=copy.deepcopy(model.encoder),
    num_samples_per_condition=32,
    generation_seed=42,
)
cond_pred_eval.run(
    dataset=mnist_test,
    output_dir=OUTPUT_DIR,
    predictor_epochs=20,
    predictor_lr=1e-3,
)
