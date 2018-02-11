import tensorflow as tf
import matplotlib.pyplot as plt
import load
import numpy as np
from PIL import Image
import pdb

# plt.plot([1, 2, 3, 4])
# plt.show()

#avg = np.mean(res, dtype=np.float64)
#std = np.std(res, dtype=np.float64)
#res = res - avg
#avg = np.mean(res, dtype=np.float64)
#res = np.divide(res, std)
#std = np.std(res, dtype=np.float64)
#print("Average: %s Standard deviation: %s"  % (avg, std))

labels = load.getLabels()
data = tf.data.Dataset.from_tensor_slices((labels))
data = data.map(load.loadImage2)
iterator = data.make_initializable_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer)
    elem = next_element.eval()
    #Image.fromarray(np.asarray(elem)).show()
    avg = np.mean(elem, dtype=np.float64)
    std = np.std(elem, dtype=np.float64)
    print("Average: %s Standard deviation: %s"  % (avg, std))
