import tensorflow as tf

import os
import pathlib
import time
import datetime

from matplotlib import pyplot as plt


from IPython import display


PATH = r"C:\Users\elias\ML\Retinal-Veinmapping\Dataset\training\gifimages"

path = pathlib.Path(PATH)


sample_image = tf.io.read_file(str(path / '21_training.gif'))
sample_image = tf.io.decode_image(sample_image, expand_animations=False)     



def load(image_file):

    image = tf.io.read_file(image_file)
    image = tf.io.decode_gif(image, expand_ainmiations=False)