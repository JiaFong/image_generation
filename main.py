import os
from os.path import join
from argparse import ArgumentParser

from src.dataset import PointColorDataset
from src.utils import load_cfg, load_trainer, scale_back, save_image, save_points
from src.evaluate import evaluate
from paths import *

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--exp", type=str)
	args = parser.parse_args()

	# Load the config file for the experiment.
	print("INFO | Loading the config file for the experiment.")
	cfg = load_cfg(args.exp)

	# Load the dataset.
	print("INFO | Loading the dataset.")
	dataset = PointColorDataset(cfg["dataset"])

	# Update the cfg with dataset-dependent settings.
	cfg["dataset"].update(dataset.extra_cfg)

	# Load the trainer.
	print("INFO | Loading the trainer.")
	trainer = load_trainer(cfg["trainer"])

	# Optimize the trainer to fit the data.
	print("INFO | Optimizing the trainer.")
	trainer.optimize(dataset)

	# Generate the image from the optimized trainer. 
	# Get those points for evaluation.
	print("INFO | Image re-generation.")
	img, generated_points = trainer.generate(cfg["dataset"]["height"],
											 cfg["dataset"]["width"],
											 cfg["dataset"]["n_points"]
											 )

	# Save the re-generated image and points.
	img_name = cfg["trainer"]["name"][:-7] + "_results.png"
	save_image(img, img_name)
	scaled_points = scale_back(generated_points, cfg["dataset"]["height"], cfg["dataset"]["width"])
	save_points(scaled_points, cfg["trainer"]["name"][:-7])


	# Evaluate the relationship between original and generated points.
	print("INFO | Evaluating the results.")
	original_points = dataset.getAllSamples()
	evaluate(original_points, generated_points, cfg["trainer"]["name"][:-7])

