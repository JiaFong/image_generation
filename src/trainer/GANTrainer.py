import os
from os.path import join
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from src.trainer.base_trainer import BaseTrainer
from paths import *

class Generator(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.model = nn.Sequential(
                nn.Linear(cfg["latent_dim"], 256),
                nn.LeakyReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 5),
                nn.Sigmoid()
            )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.model = nn.Sequential(
                nn.Linear(cfg["in_dim"], 256),
                nn.LeakyReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        return self.model(x)

class GANTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.generator = Generator(self.cfg["generator"])
        self.discriminator = Discriminator(self.cfg["discriminator"])

    def load(self):
        path = join(SAVED_PATH, self.cfg["name"][:-7] + ".ckpt")
        loader = torch.load(path, map_location=self.cfg["device"])
        self.generator.load_state_dict(loader["generator"])
        self.discriminator.load_state_dict(loader["discriminator"])

    def save(self):
        if not os.path.exists(SAVED_PATH):
            os.mkdir(SAVED_PATH)
        path = join(SAVED_PATH, self.cfg["name"][:-7] + ".ckpt")
        torch.save({"generator": self.generator.state_dict(),
                    "discriminator": self.discriminator.state_dict()}, path)

    def optimize(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.cfg["batch_size"], shuffle=True)

        criterion = nn.BCELoss()
        optimizer_G = AdamW(self.generator.parameters(),
                            lr=self.cfg["lr"],
                            weight_decay=self.cfg["weight_decay"])
        optimizer_D = AdamW(self.discriminator.parameters(),
                            lr=self.cfg["lr"],
                            weight_decay=self.cfg["weight_decay"])

        self.generator.to(self.cfg["device"])
        self.discriminator.to(self.cfg["device"])

        self.generator.train()
        self.discriminator.train()
        for epoch in range(self.cfg["epoch"]):
            total_g_loss, total_d_loss = 0, 0
            for real_samples in dataloader:
                batch_size = real_samples.size(0)
                latent_dim = self.cfg["generator"]["latent_dim"]
                
                real_samples = real_samples.to(self.cfg["device"])
                real_labels = torch.ones(batch_size, 1, device=self.cfg["device"])
                fake_labels = torch.zeros(batch_size, 1, device=self.cfg["device"])

                # Train Discriminator
                z = torch.randn(batch_size, latent_dim, device=self.cfg["device"])
                fake_samples = self.generator(z)
                
                d_loss_real = criterion(self.discriminator(real_samples), real_labels)
                d_loss_fake = criterion(self.discriminator(fake_samples.detach()), fake_labels)
                d_loss = d_loss_real + d_loss_fake
                
                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()
                total_d_loss += d_loss.cpu().detach().item()

                # Train Generator
                z = torch.randn(batch_size, latent_dim, device=self.cfg["device"])
                fake_samples = self.generator(z)
                g_loss = criterion(self.discriminator(fake_samples), real_labels)

                optimizer_G.zero_grad()
                g_loss.backward()
                optimizer_G.step()

                total_g_loss += g_loss.cpu().detach().item()
            
            print(f"Epoch {epoch+1}, Discriminator loss: {total_d_loss/len(dataloader):.4f}")
            print(f"Epoch {epoch+1}, Generator loss: {total_g_loss/len(dataloader):.4f}")

    def sample(self, n_points):
        self.generator.to(self.cfg["device"])
        self.generator.eval()

        with torch.no_grad():
            z = torch.randn(5000, self.cfg["generator"]["latent_dim"]).to(self.cfg["device"])
            samples = self.generator(z)
            points = samples.cpu().detach().numpy()
        return points