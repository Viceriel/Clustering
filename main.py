import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import load
import pdb
import numpy as np

# plt.plot([1, 2, 3, 4])
# plt.show()
res = load.loadImage()
avg = np.mean(res, dtype=np.float64)
std = np.std(res, dtype=np.float64)
res = res - avg
avg = np.mean(res, dtype=np.float64)
res = np.divide(res, std)
std = np.std(res, dtype=np.float64)
print("Average: %s Standard deviation: %s"  % (avg, std))
