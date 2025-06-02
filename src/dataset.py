from os.path import join
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from paths import *

class PointColorDataset(Dataset):
	def __init__(self, cfg):
		self.device = cfg["device"]

		# loading and preprocess data
		xPath = join(DATASET_PATH, "pi_xs.npy")
		yPath = join(DATASET_PATH, "pi_ys.npy")
		xs = np.load(xPath)
		ys = np.load(yPath)
		
		imgPath = join(DATASET_PATH, "sparse_pi_colored.jpg")
		img = np.array(Image.open(imgPath)).astype(np.float32)
		rgbs = img[xs, ys] / 255.0

		# build the dataset of 5D sample (x, y, r, g, b)
		# all range in [0.0, 1.0]
		height, width = img.shape[0], img.shape[1]
		self.data = np.concatenate([(xs / height).reshape(-1, 1), 
									(ys / width).reshape(-1, 1), 
									rgbs], axis=1)

		self.extra_cfg = {"height": height,
						  "width": width,
						  "n_points": xs.shape[0]
						}
	def __len__(self):
		return self.data.shape[0]
		
	def __getitem__(self, index):
		return torch.tensor(self.data[index]).float().to(self.device)

	def getAllSamples(self, return_in="numpy"):
		if return_in == "numpy":
			return self.data
		else:
			return torch.tensor(self.data).float().to(self.device)