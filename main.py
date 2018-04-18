import tensorflow as tf
import matplotlib.pyplot as plt
import load
from kmeans import KMeans

pattern= "./real/Jpeg8bit/*.jpg"
#pattern= "./real/Jpeg8bit/*.bmp"
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
plt.figure()
plt.ion()
plt.imshow(img)
plt.colorbar()
plt.show() 
plt.pause(0.001)

def input_fn():
  return tf.train.limit_epochs(
      tf.convert_to_tensor(points, dtype=tf.float32), num_epochs=1)

points = elem.flatten();
points = load.adjustDimension(points)
kmeans = tf.contrib.factorization.KMeansClustering(num_clusters=6, use_mini_batch=False)
num_iterations = 8
previous_centers = None
for i in range(num_iterations):
  kmeans.train(input_fn)
  cluster_centers = kmeans.cluster_centers()
  if previous_centers is not None:
    print ('delta:',cluster_centers - previous_centers)
  previous_centers = cluster_centers
  print ('score:', kmeans.score(input_fn))
print ('cluster centers:', cluster_centers)

clusters = KMeans(elem, 6)
clusters.centeroids = [cluster_centers[0][1], cluster_centers[1][1], cluster_centers[2][1], cluster_centers[3][1], cluster_centers[4][1], cluster_centers[5][1]];
img = clusters.printImg();
plt.figure()
plt.ion()
plt.imshow(img)
plt.colorbar()
plt.show() 
plt.pause(0.001)
print("end")
