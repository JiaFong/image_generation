# Image Generation Project

## Introduction

### Task
This repository aims to address a density modeling and sample generation problem over a 5D space:

	* Spatial: image coordinates (x, y) range in [0, 299]
	* Color: pixel’s RGB values (r, g, b) range in [0, 255]

The objective of the task is to learn a distribution over colored points such that samples drawn from the model exhibit the same spatial and color characteristics as those in the original dataset.

## How to Run the Project

1. Download the project via git clone.
```
git clone https://github.com/JiaFong/image_generation.git
```

2. Create a new environment via conda.
```
conda create -n YOUR_ENV_NAME python=3.9.21
```
Following the instructions to build the environment.

3. Activate the environment and install the required package in requirements.txt.
```
conda activate YOUR_ENV_NAME
pip3 install -r requirements.txt
```

3. Run the main.py with the targeted experiment. Check the config files of supported experiments in the config folder for more options.
```
python main.py --exp train_gmm
```
The above code will train a GMM model and re-sample the points.
The generated points will be used for the image generation and performance evaluations.

4. You can create your own generative methods and experiments. Remember to inherent the BaseTrainer class and create the config file for the new methods. 

## Results
If the program is successfully executed, you should see the outputs similar like this:

![execution results](https://github.com/user-attachments/assets/75601607-d35e-449e-b991-000165d48a8f)

We evaluate how suprise the original and re-generated points may be sampled from a similar distribution by the following metrics:
- Feature visualization (seaborn's pairplot)
- Maximum Mean Discrepancy (MMD)
- KL Divergence via kernel density estimation (KDE)

The MMD and KL score will be shown in the command line; and you can find the re-generated image and pairplot results in the saved folder. Below are the results from the GMM method.

![re-generated image](https://github.com/user-attachments/assets/75601607-d35e-449e-b991-000165d48a8f)

![feature visualization](https://github.com/user-attachments/assets/75601607-d35e-449e-b991-000165d48a8f)
