import tensorflow as tf

import os
import pathlib
import time
import datetime

from matplotlib import pyplot as plt


from IPython import display


PATH = r"C:\Users\elias\ML\Retinal-Veinmapping\Dataset\training"


epath = pathlib.Path(EPATH)
rpath = pathlib.Path(RPATH)



def load(num): # returns tensor of image

    real = tf.io.read_file(f"{path}\gifimages/{num}_training.gif")
    real = tf.io.decode_image(real, expand_animations=False)

    target = tf.io.read_file(fr"{path}\1st_manual/{num}_manual1.gif")
    target = tf.io.decode_image(target, expand_animations=False)

    return tf.cast(real, tf.float32), tf.cast(target, tf.float32)


print(load(21)[0].shape)

BUFFER_SIZE = 20

BATCH_SIZE = 1

IMG_WIDTH = 565
IMG_HEIGHT = 584



def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image, real_image


def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]


# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image):
  # Resizing to 286x286
  input_image, real_image = resize(input_image, real_image, 286, 286)

  # Random cropping back to 256x256
  input_image, real_image = random_crop(input_image, real_image)

  if tf.random.uniform(()) > 0.5:
    # Random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image


inp, re = load(21)
print(inp.shape, re.shape)

plt.figure(figsize=(6, 6))
for i in range(4):
  rj_inp, rj_re = random_jitter(inp, re)
  plt.subplot(2, 2, i + 1)
  plt.imshow(rj_inp / 255.0)
  plt.axis('off')
plt.show()