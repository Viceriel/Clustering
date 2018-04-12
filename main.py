import tensorflow as tf
import matplotlib.pyplot as plt
import load
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

clusters = KMeans(elem, 6)
img = clusters.clustering(5)
plt.imshow(img)
plt.colorbar()
plt.show()
print("end")
