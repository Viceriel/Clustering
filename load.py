import tensorflow as tf
import numpy as np

def loadImage(img_path):
    """Converts image at path to normalised tensor"""
    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_image(img_file)

    img_decoded=tf.reshape(img_decoded, [512, 512])
    #img_decoded = tf.reshape(tf.div(tf.to_float(img_decoded), 255), [-1])
    return img_decoded

def getLabels(pattern):
    """Returning array with names of jpeg images at images folder"""
    pattern = tf.train.match_filenames_once(pattern)
    init_op = tf.local_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        labels = pattern.eval()

    return labels

def adjustDimension(arr):
    length = len(arr)
    out = []

    for i in range(0, length):
        if (arr[i] != 0):
            out.append([0, arr[i]])

    return out
