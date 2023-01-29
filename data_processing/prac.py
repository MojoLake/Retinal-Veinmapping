import tensorflow as tf

import os
import pathlib
import time
import datetime

from matplotlib import pyplot as plt


from IPython import display


EPATH = r"C:\Users\elias\ML\Retinal-Veinmapping\Dataset\training\gifimages"
RPATH = r"C:\Users\robin\OneDrive\Työpöytä\Retinas\Retinal-Veinmapping\Dataset\training\gifimages"

epath = pathlib.Path(EPATH)
rpath = pathlib.Path(RPATH)

sample_image = tf.io.read_file(str(rpath / '21_training.gif'))
sample_image = tf.io.decode_image(sample_image, expand_animations=False)     


def load(image_file):

    image = tf.io.read_file(image_file)
    image = tf.io.decode_image(image, expand_animations=False)
    return(image)


img = load(str(rpath / '21_training.gif'))

plt.figure()
plt.imshow(img)
plt.show()



