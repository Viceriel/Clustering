import tensorflow as tf
import numpy as np

def loadImage2(img_path):
    """Converts image at path to normalised tensor"""
    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_image(img_file)

    img_decoded = tf.reshape(tf.div(tf.to_float(img_decoded), 255), [-1])
    return img_decoded

def getLabels():
    """Returning array with names of jpeg images at images folder"""
    pattern = tf.train.match_filenames_once("./images/*.jpg")
    init_op = tf.local_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        labels = pattern.eval()

    return labels
