import tensorflow as tf
import matplotlib.pyplot as plt
import load
import pdb
from kmeans import KMeans

pattern= "./real/Jpeg8bit/*.jpg"
labels = load.getLabels(pattern)
print(len(labels))
data = tf.data.Dataset.from_tensor_slices((labels))
data = data.map(load.loadImage)
iterator = data.make_initializable_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer)
    while True:
        try:
            elem = next_element.eval()
            next_element = iterator.get_next()
        except tf.errors.OutOfRangeError:
            break

k = KMeans(elem, 3)
img = k.clustering(5)
pdb.set_trace()
plt.imshow(img)
plt.colorbar()
plt.show()
print("end")
