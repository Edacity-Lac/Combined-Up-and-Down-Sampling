import math
import numpy as np


def psnr(img1, img2):
	mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
	psnr = 20 * math.log10(1 / math.sqrt(mse))
	return psnr

def mse(img1, img2):
	mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
	return mse
