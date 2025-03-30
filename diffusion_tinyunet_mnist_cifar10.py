import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import os
import matplotlib.pyplot as plt
import numpy as np


# U-Net with multi-level skip connections for MNIST/CIFAR10
class UNetDenoiser(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(channels + 1, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU()
        )
        self.middle = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU()
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU()
        )
        self.out = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, channels, 3, padding=1)
        )

    def forward(self, x, t):
        t_embed = t.view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        xt = torch.cat([x, t_embed], dim=1)
        x1 = self.down1(xt)         # [B, 64, H, W]
        x2 = self.down2(x1)         # [B, 128, H/2, W/2]
        x_mid = self.middle(x2)     # [B, 128, H/2, W/2]
        x_up = self.up1(x_mid)      # [B, 64, H, W]
        x_cat = torch.cat([x_up, x1], dim=1)  # Skip connection from down1
        out = self.out(x_cat)
        return out


# Define a simple linear noise schedule
T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1. - betas
alpha_hat = torch.cumprod(alphas, dim=0)

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu()).reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    return out

def add_noise(x, t, noise):
    sqrt_alpha_hat = get_index_from_list(torch.sqrt(alpha_hat), t, x.shape)
    sqrt_one_minus_alpha_hat = get_index_from_list(torch.sqrt(1 - alpha_hat), t, x.shape)
    return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise


# -----------------------------
# Training Function
# -----------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "mnist":
        channels = 1
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    elif args.dataset == "cifar10":
        channels = 3
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = UNetDenoiser(channels=channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("Dataloader, Model and Optimizer setup complete!")
    print("Training has started ...")
    model.train()
    for epoch in range(args.epochs):
        print(f"Running Epoch: {epoch}")
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)
            t = torch.randint(0, T, (x.size(0),), device=device).long()
            noise = torch.randn_like(x)
            x_noisy = add_noise(x, t, noise)
            noise_pred = model(x_noisy, t)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % args.log_interval == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Batch [{i}] Loss: {loss.item():.4f}")

    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"./checkpoints/diffusion_{args.dataset}.pt")
    print("Training complete.")


# -----------------------------
# Inference / Sampling Function
# -----------------------------
def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "mnist":
        channels, img_size = 1, 28
    elif args.dataset == "cifar10":
        channels, img_size = 3, 32
    else:
        raise ValueError("Unsupported dataset")

    model = UNetDenoiser(channels).to(device)
    model.load_state_dict(torch.load(f"./checkpoints/diffusion_{args.dataset}.pt", map_location=device))
    model.eval()
    print("Model setup for inference")
    print("Inference running")

    os.makedirs("./samples", exist_ok=True)

    with torch.no_grad():
        x = torch.randn(args.num_samples, channels, img_size, img_size).to(device)
        history = []
        for t in reversed(range(T)):
            t_batch = torch.full((args.num_samples,), t, device=device, dtype=torch.long)
            beta_t = get_index_from_list(betas, t_batch, x.shape)
            alpha_t = get_index_from_list(alphas, t_batch, x.shape)
            alpha_hat_t = get_index_from_list(alpha_hat, t_batch, x.shape)

            noise_pred = model(x, t_batch)

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * noise_pred) + torch.sqrt(beta_t) * noise

            if t % (T // 10) == 0:
                history.append(x.clone().cpu())

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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--inference", action="store_true", help="Run inference after training")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to generate in inference")
    args = parser.parse_args()

    train(args)
    if args.inference:
        inference(args)

