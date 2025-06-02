import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import entropy, gaussian_kde
from src.utils import save_image

def evaluate(ori_points, gen_points, method_name):
    # Compare them by feature visualiztion (seaborn pairplot).
    print("INFO | Run the pairwise feature visualization; checking the saved directory for the result.")
    pfv_img = pairwise_feature_visualization(ori_points, gen_points)
    save_image(pfv_img, method_name+"_pairplot.png")
    # Compare them by Maximum Mean Discrepancy (MMD) -> lower is better.
    print("INFO | Compute the Maximum Mean Discrepancy score.")
    mmd_score = compute_maximum_mean_discrepancy(ori_points, gen_points)
    print("INFO | MMD score : {}".format(mmd_score))

    print("INFO | Compute the KL divergence score.")
    kl_score = compute_kl_divergence(ori_points, gen_points)
    print("INFO | KL divergence score : {}".format(kl_score))

def pairwise_feature_visualization(ori_points, gen_points):
    df_ori = pd.DataFrame(ori_points, columns=["x", "y", "r", "g", "b"])
    df_gen = pd.DataFrame(gen_points, columns=["x", "y", "r", "g", "b"])
    df_ori["label"] = "original"
    df_gen["label"] = "generated"
    df = pd.concat([df_ori, df_gen])
    img = sns.pairplot(df, hue="label", plot_kws = {"alpha": 0.3})
    return img

def compute_maximum_mean_discrepancy(ori_points, gen_points):
    # Set the gamma by Median Heuristic.
    Z = np.vstack([ori_points, gen_points])
    dists = pairwise_distances(Z, Z)
    median_dist = np.median(dists)
    gamma = 1 / (2 * median_dist**2)

    # Compute the mmd score.
    K_oo = rbf_kernel(ori_points, ori_points, gamma=gamma)
    K_oa = rbf_kernel(ori_points, gen_points, gamma=gamma)
    K_aa = rbf_kernel(gen_points, gen_points, gamma=gamma)
    return K_oo.mean() + K_aa.mean() - 2 * K_oa.mean()


def compute_kl_divergence(ori_points, gen_points):
    # Estimate the distribution by gaussian_kde.
    p_kde = gaussian_kde(ori_points.T)
    q_kde = gaussian_kde(gen_points.T)

    # Use original points to get the density values.
    xs = ori_points.T
    p_vals = p_kde(xs)
    q_vals = q_kde(xs)

    return entropy(p_vals, q_vals) # forward KL