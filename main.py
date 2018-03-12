import tensorflow as tf
import matplotlib.pyplot as plt
import load
import numpy as np
from PIL import Image
import pdb
import kmeans
import cv2


# plt.plot([1, 2, 3, 4])
# plt.show()

#real = input("Do you want use test(0) or real(1) dataset: ")

#if real == "1":
pattern= "./real/Jpeg8bit/*.jpg"
#else:
    #pattern= "./images/*.jpg"

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
            #Image.fromarray(np.asarray(elem)).show()
            next_element = iterator.get_next()
        except tf.errors.OutOfRangeError:
            break
    #print(np.cov(elem))

k = kmeans.KMeans(elem, 3)
img = k.clustering(5)
import pdb
pdb.set_trace()
#Image.fromarray(np.asarray(img)).show()
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.colorbar()
plt.show()
print("end")
