import copy
import numpy as np
from collections import defaultdict
from sklearn.mixture import GaussianMixture
from src.dataset import PointColorDataset
from src.trainer.base_trainer import BaseTrainer
from PIL import Image

class GMMTrainer(BaseTrainer):
	def __init__(self, cfg):
		super().__init__(cfg)

		self.model = None

	def optimize(self, dataset):
		x = dataset.getAllSamples(return_in="numpy")

		best_bic_k, best_bic = -1, np.inf
		best_score_k, best_score = -1, -np.inf
		min_k, max_k = self.cfg["min_k"], self.cfg["max_k"]
		for k in range(min_k, max_k+1):
			gmm = GaussianMixture(n_components=k, covariance_type='diag', random_state=0, init_params="k-means++")
			gmm.fit(x)

			bic = gmm.bic(x)
			score = gmm.score(x)
			if score > best_score:
				best_score_k = k
				best_score = score

			if bic < best_bic:
				best_bic_k = k
				best_bic = bic
				self.model = copy.deepcopy(gmm)

		print("INFO | After optimization, best bic and score locate at {} and {}, respectively.".format(best_bic_k, best_score_k))

	def sample(self, n_points):
		points, _ = self.model.sample(n_points)
		return points













		

