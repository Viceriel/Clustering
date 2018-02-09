import tensorflow as tf
import numpy as np
from PIL import Image

def loadImage():
    pattern = tf.train.match_filenames_once("./images/*.jpg")
    filename_queue = tf.train.string_input_producer(pattern)
    #filename_queue = tf.train.string_input_producer(["./images/dom1.jpg"])
    image_reader = tf.WholeFileReader()
    _, image_file = image_reader.read(filename_queue)
    my_img = tf.image.decode_jpeg(image_file)

    init_op = tf.local_variables_initializer()
    image = []

    with tf.Session() as sess:
        sess.run(init_op)
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        num = len(pattern.eval())

        for i in range(num): #length of your filename list
            image.append(my_img.eval()) #here is your image Tensor :)

            print(image[i].shape)
            Image.fromarray(np.asarray(image[i])).show()

            coord.request_stop()
            coord.join(threads)

    return image
