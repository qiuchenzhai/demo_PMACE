import numpy as np
import matplotlib.pyplot as plt
from math import *
from pylab import *


def calculate_nrmse(x, y, cstr):
    """
    The Normalized Root-Mean-Square-Error (NRMSE) between two images x and y.
    :param x: the complex recovery.
    :param y: the complex ground truth image.
    :return: NRMSE between x and y.
    """
    # Initialization
    num_pixels = float(np.sum(cstr))
    y = cstr * y
    x = cstr * x
    # Compute MSE
    mse = np.sum(np.abs(x - y) ** 2) / num_pixels
    # rmse = np.sqrt(mse)
    temp = np.sqrt(np.sum(np.abs(y) ** 2) / num_pixels)
    nrmse = np.divide(np.sqrt(mse), temp, where=(temp != 0))

    return nrmse


def pha_err(gt, img):
    """
    The phase error = min | angle(img) - angle(gt) - 2*k*pi| where k \in {-1, 0, 1}
    :param gt: the ground truth image.
    :param img: the recovered image.
    :return: the phase error between the ground truth image and recovered image.
    """
    ang_diff = np.angle(gt) - np.angle(img)
    ang_diff[ang_diff > pi] -= pi
    ang_diff[ang_diff < -pi] += pi
    pha_error = np.minimum(ang_diff, ang_diff + 2*pi)

    return pha_error


def phase_norm(img, gt):
    """
    The reconstruction is blind to absolute phase of ground truth image, so need to make
    phase shift to the reconstruction results given the known ground truth image.
    :param img: the reconstruction needs phase normalization.
    :param gt: the known ground truth image.
    :return: the phase normalized reconstruction.
    """
    # phase normalization
    img_size = np.shape(img)
    img = np.reshape(img, -1)
    gt = np.reshape(gt, -1)
    norm_term = np.divide(np.sum(np.dot(np.conj(img), gt)), np.abs(np.sum(np.dot(np.conj(img), gt))),
                          where=(np.abs(np.sum(np.dot(np.conj(img), gt))) != 0))
    img = norm_term * img
    img.resize(img_size)

    return img

