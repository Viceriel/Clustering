import tensorflow as tf
import matplotlib.pyplot as plt
import load
import numpy as np
from PIL import Image
import pdb

# plt.plot([1, 2, 3, 4])
# plt.show()

labels = load.getLabels()
data = tf.data.Dataset.from_tensor_slices((labels))
data = data.map(load.loadImage2)
iterator = data.make_initializable_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer)
    elem = next_element.eval()
    #Image.fromarray(np.asarray(elem)).show()
    print(elem)
