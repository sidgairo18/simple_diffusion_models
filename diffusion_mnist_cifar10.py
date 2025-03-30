"""
A simple diffusion model code walkthrough on MNIST and CIFAR-10 datasets.
"""
import os
import argparse
# torch based imports below
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
# misc imports
import matplotlib.pyplot as plt
import numpy as np

class SimpleDenoiser(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
                nn.Conv2d(channels+1, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, channels, 3, padding=1)
                )

    def forward(self, x, t):
        t_embed = t.view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        x = torch.cat([x, t_embed], dim=1)
        return self.net(x)

def add_noise(x, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x)
    while len(t.shape) < len(x.shape):
        t = t.unsqueeze(-1)
    return torch.sqrt(1 - t) * x + torch.sqrt(t) * noise

# -----------------------------
# Training Function
# -----------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"The model will run on device: {device}")

    # Data loader setup
    if args.dataset == "mnist":
        channels = 1
        ransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            ])
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    elif args.dataset == 'cifar10':
        channels = 3
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset! Supported datasets are [cifar10, mnist]")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model, optimizer
    model = SimpleDenoiser(channels=channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(dataloader))

    model.train()
    print("Training has begun ...")
    for epoch in range(args.epochs):
        print(f"Epoch : {epoch} running ...")
        for i, (x, _) in enumerate(dataloader):
            #print("running", i)
            x = x.to(device)
            t = torch.rand(x.size(0), device=device)
            noise = torch.randn_like(x)
            x_noisy = add_noise(x, t, noise)

            noise_pred = model(x_noisy, t)
            loss = F.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i%args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Batch [{i+1}], Loss: {loss.item():.4f}")

    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"./checkpoints/diffusion_{args.dataset}.pt")
    print("Training complete.")

# -----------------------------
# Inference / Sampling Function
# -----------------------------
def inference(args):
    print("Running inference ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if args.dataset == "mnist":
        channels, img_size = 1, 28
    elif args.dataset == 'cifar10':
        channels, img_size = 3, 32
    else:
        raise ValueError("Unsupported dataset. Supported datasets are [cifar10, mnist]")

    model = SimpleDenoiser(channels).to(device)
    model.load_state_dict(torch.load(f"./checkpoints/diffusion_{args.dataset}.pt", map_location=device))
    model.eval()  # Set the model to evaluation mode

    os.makedirs("./samples", exist_ok=True)

    with torch.no_grad():
        x = torch.randn(args.num_samples, channels, img_size, img_size).to(device)
        steps = 500
        history = []
        for step in reversed(range(1, steps+1)):
            t = torch.full((args.num_samples,), step / steps, device=device)
            noise_pred = model(x, t)
            x = x - noise_pred * 0.1 # basic denoising step
            if step %(steps // 5) == 0:
                history.append(x.clone().cpu())

        # Plot the denoising process
        for i, stage in enumerate(history):
            grid = utils.make_grid(stage, nrow=int(np.sqrt(args.num_samples)), normalize=True)
            plt.figure(figsize=(6, 6))
            plt.imshow(grid.permute(1, 2, 0))
            plt.title(f"Denoising step {i+1}")
            plt.axis("off")
            plt.savefig(f"./samples/{args.dataset}_step_{i+1}.png")
            #plt.show()
            plt.close()

# -----------------------------
# Argument Parser
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal Diffusion Model for MNIST/CIFAR-10")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--inference", action="store_true", help="Run inference after training")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to generate in inference")
    args = parser.parse_args()
    
    if not args.inference:
        train(args)
    if args.inference:
        inference(args)




