import numpy as np
from abc import ABC, abstractmethod
from PIL import Image
from collections import defaultdict

class BaseTrainer(ABC):
	def __init__(self, cfg):
		self.cfg = cfg

		@property
		@abstractmethod
		def model(self):
			raise NotImplementedError

	@abstractmethod
	def optimize(self, dataset):
		raise NotImplementedError

	@abstractmethod
	def sample(self, n_points):
		raise NotImplementedError

	def generate(self, height=300, width=300, n_points=5000):
		# Create the canvas for drawing those sampled points
		canvas = np.zeros([height, width, 3], dtype=np.uint8)
		
		# Sample n_points points for drawing
		points = self.sample(n_points)
		
		# Get all location x and y, making sure their values \ 
		# are in the canvas
		xs = np.clip((height * points[:, 0]).astype(int), 0, height-1)
		ys = np.clip((width * points[:, 1]).astype(int), 0, width-1)

		# Scaling the RGB values back, but without clipping or casting. \
		# We need to deal with the case where the same location (x, y) has \
		# multiple RGB values
		rgbs = points[:, 2:] * 255.0

		pixel_dict = defaultdict(list)
		for x, y, rgb in zip(xs, ys, rgbs):
			pixel_dict[(x, y)].append(rgb)

		# For every location which has rgb value(s), we compute the \
		# averaged RGB value and set the alpha to non-transparent
		for (x, y), rgb in pixel_dict.items():
			avg_rgb = np.mean(rgb, axis=0)
			avg_rgb = np.clip(avg_rgb, 0, 255).astype(np.uint8)
			canvas[x, y, :3] = avg_rgb

		# Create and return the image in PIL.
		# The sampled points are also return for advanced usages
		img = Image.fromarray(canvas)
		return img, points
		 