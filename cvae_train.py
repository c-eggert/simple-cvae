from typing import Tuple

import torch
from torch import nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST

import cvae
from cvae import ConditionSchema, ConditionVariable, InputEncoderCategoricalToOneHot
from cvae.cvae import DecoderBase, UpConvBlock, EncoderBase, ConvBlock


class Encoder(EncoderBase):
    def __init__(self, in_channels_data: int, in_channels_cond: int, out_channels: int):
        super(Encoder, self).__init__()
        self.conv1 = ConvBlock(
            in_channels_data=in_channels_data,
            in_channels_cond=in_channels_cond,
            out_channels=16,
            num_groups=4,
            stride=1,
            padding=1,
        )
        self.conv2 = ConvBlock(
            in_channels_data=16,
            in_channels_cond=in_channels_cond,
            out_channels=32,
            num_groups=8,
            stride=2,
            padding=1,
        )
        self.conv3 = ConvBlock(
            in_channels_data=32,
            in_channels_cond=in_channels_cond,
            out_channels=64,
            num_groups=8,
            stride=2,
            padding=1,
        )
        self.conv4 = ConvBlock(
            in_channels_data=64,
            in_channels_cond=in_channels_cond,
            out_channels=128,
            num_groups=8,
            stride=1,
            padding=0,
        )
        self.latent_mu = nn.Linear(128, out_channels)
        self.latent_logvar = nn.Linear(128, out_channels)

    def forward(
        self, data: torch.Tensor, condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(data, condition)
        x = self.conv2(x, condition)
        x = self.conv3(x, condition)
        x = self.conv4(x, condition)
        x = torch.squeeze(x, dim=[-2, -1])
        mu = self.latent_mu(x)
        logvar = self.latent_logvar(x)
        return mu, logvar


class Decoder(DecoderBase):
    def __init__(self, in_latent_dim: int, in_channels_cond: int, out_channels: int):
        super(Decoder, self).__init__()
        self.unproject = nn.Linear(in_latent_dim, 128)
        self.up1 = UpConvBlock(128, in_channels_cond, 64, 8, 1, padding=1)
        self.up2 = UpConvBlock(64, in_channels_cond, 32, 8, 1, padding=1)
        self.up3 = UpConvBlock(32, in_channels_cond, 16, 8, 1, padding=0)
        self.up4 = UpConvBlock(16, in_channels_cond, 8, 4, 1, padding=1)
        self.pred = nn.Conv2d(in_channels=8, out_channels=out_channels, kernel_size=1)

    def forward(self, sampled: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = self.unproject(sampled)
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        x = self.up1(x, condition)
        x = self.up2(x, condition)
        x = self.up3(x, condition)
        x = self.up4(x, condition)
        x = self.pred(x)
        return x


condition_schema = ConditionSchema([
    ConditionVariable("digit", InputEncoderCategoricalToOneHot(num_classes=10)),
])


transform_image = transforms.Compose([
    transforms.Resize((11, 11), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.float()),
])
transform_label = transforms.Lambda(lambda y: condition_schema.encode({"digit": y}))

mnist_train = MNIST(root="data", train=True,  download=False, transform=transform_image, target_transform=transform_label)
mnist_test  = MNIST(root="data", train=False, download=False, transform=transform_image, target_transform=transform_label)

loader_train = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True)
loader_test  = torch.utils.data.DataLoader(mnist_test,  batch_size=64, shuffle=False)

LATENT_DIM = 16
COND_DIM = condition_schema.output_dim

encoder = Encoder(in_channels_data=1, in_channels_cond=COND_DIM, out_channels=LATENT_DIM)
decoder = Decoder(in_latent_dim=LATENT_DIM, in_channels_cond=COND_DIM, out_channels=1)
model = cvae.CVAE(encoder=encoder, decoder=decoder)


trainer = cvae.Trainer(
    model=model,
    train_loader=loader_train,
    val_loader=loader_test,
    kl_weight=0.001,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    device=None,
)
trainer.fit(epochs=200)
