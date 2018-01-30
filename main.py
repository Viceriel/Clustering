import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pdb

# plt.plot([1, 2, 3, 4])
# plt.show()
pdb.set_trace()
filename_queue = tf.train.string_input_producer(["./images/dom1.jpg"])
image_reader = tf.WholeFileReader()
_, image_file = image_reader.read(filename_queue)
my_img = tf.image.decode_jpeg(image_file)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    # Start populating the filename queue.

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(1): #length of your filename list
        image = my_img.eval() #here is your image Tensor :)

    print(image.shape)
    Image.fromarray(np.asarray(image)).show()

    coord.request_stop()
    coord.join(threads)
