import tensorflow as tf

import os
import pathlib
import time
import datetime

from skimage import io
from matplotlib import pyplot as plt


from IPython import display


PATH = r"C:\Users\elias\ML\Retinal-Veinmapping\Dataset"

path = pathlib.Path(PATH)



def load(image_file): # returns tensor of image

    # print("IMAGE FILE:", image_file)
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    # plt.figure()
    # plt.imshow(image / 255.0)
    # plt.show()


    return image



BUFFER_SIZE = 20

BATCH_SIZE = 1

IMG_WIDTH = 256
IMG_HEIGHT = 256



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
#   input_image, real_image = resize(input_image, real_image, 286, 286)
    # these twom make some of the picture blue for some reason
  # Random cropping back to 256x256
#   input_image, real_image = random_crop(input_image, real_image)

 

  if tf.random.uniform(()) > 0.5:
    # Random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  if tf.random.uniform(()) > 0.5:

    input_image = tf.image.flip_up_down(input_image)
    real_image = tf.image.flip_up_down(real_image)

  return input_image, real_image


inp = load(str(path / r'training\images_256\22_train.jpg'))
re = load(str(path / r'training\1st_manual_256\22_manual.jpg'))
# plt.figure()
# plt.imshow(inp / 255.0)
# plt.figure()
# plt.imshow(re / 255.0)
# plt.show()



def load_image_train(image_file):
  return load(image_file)


def create_more(input_image, real_image):
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image



def load_image_test(image_file):
  input_image = load(image_file)
  input_image, _ = normalize(input_image, input_image)

  return input_image