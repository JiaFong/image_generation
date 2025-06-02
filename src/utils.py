import os
from os.path import join
import yaml
import importlib
import numpy as np
import seaborn as sns
import pandas as pd
from PIL import Image
from src.trainer import SUPPORTED_TRAINER
from paths import *

def yaml_loader(file):
    with open(file, "r") as fp:
        data = yaml.load(fp, Loader=yaml.FullLoader)
    return data

def load_cfg(exp_name):
	path = join(CONFIG_PATH, exp_name+".yaml")

	if not os.path.exists(path):
		raise IOError("ERRO | Can't find the config file for the experiment: {}.".format(exp_name))
	else:
		return yaml_loader(path)

def load_trainer(cfg):
	trainer_name = cfg["name"]

	if not trainer_name in SUPPORTED_TRAINER:
		raise NotImplementedError("ERRO | {} is not a supported trainer.".format(trainer_name))

	src = (TRAINER_PATH + trainer_name).replace(ROOT_PATH, "")
	src = src.replace("/", ".")

	module = importlib.import_module(src)
	return getattr(module, trainer_name)(cfg)

def scale_back(points, height, width):
	max_vals = np.array([height, width, 255, 255, 255], dtype=int)

	scaled_points = points * max_vals
	return np.clip(scaled_points, np.zeros(5), max_vals)

def save_points(points, method_name):
	if not os.path.exists(SAVED_PATH):
		os.mkdir(SAVED_PATH)

	path = join(SAVED_PATH, method_name+"_points.npy")
	with open(path, "wb") as f:
		np.save(f, points)

def save_image(img, img_name):
	if not os.path.exists(SAVED_PATH):
		os.mkdir(SAVED_PATH)

	path = join(SAVED_PATH, img_name)

	if isinstance(img, Image.Image):
		img.save(path)
	else:
		img.figure.savefig(path, dpi=300, bbox_inches='tight')