import torch
from torchvision import transforms
from torchvision.datasets import MNIST
import cvae


transform_image = transforms.Compose([
    transforms.Resize((11, 11), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.float())])
transform_label = transforms.Compose([
    cvae.InputEncoderCategoricalToOneHot(num_classes=10, dtype=torch.int32),
    transforms.Lambda(lambda x: x.float())])


mnist_train = MNIST(root='data', train=True, download=False, transform=transform_image, target_transform=transform_label)
mnist_test = MNIST(root='data', train=False, download=False, transform=transform_image, target_transform=transform_label)

loader_train = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True)
loader_test = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=False)

model = cvae.CVAE(
    in_channels_data=1,
    in_channels_cond=10,
    latent_channels=16
)

trainer = cvae.Trainer(
    model=model,
    train_loader=loader_train,
    val_loader=loader_test,
    kl_weight=0.001,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    device=None
)
trainer.fit(epochs=200)
