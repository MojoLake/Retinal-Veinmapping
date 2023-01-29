import tensorflow as tf

import os
import pathlib
import time
import datetime

from matplotlib import pyplot as plt


from IPython import display


PATH = r"C:\Users\elias\ML\Retinal-Veinmapping\Dataset\training"


path = pathlib.Path(PATH)



def load(num): # returns tensor of image

    real = tf.io.read_file(fr"{path}\jpgimages/{num}_training.jpg")
    real = tf.io.decode_jpeg(real)

    target = tf.io.read_file(fr"{path}\1st_manualjpg/{num}_manual1-0000.jpg")
    target = tf.io.decode_jpeg(target)
    target = tf.image.grayscale_to_rgb(target,name=None)

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
  input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)

  # Random cropping back to 256x256
  input_image, real_image = random_crop(input_image, real_image)

 

  if tf.random.uniform(()) > 0.5:
    # Random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  if tf.random.uniform(()) > 0.5:

    input_image = tf.image.flip_up_down(input_image)
    real_image = tf.image.flip_up_down(real_image)

  return input_image, real_image


inp, re = load(21)



def load_image_train(num):
  input_image, real_image = load(num)
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image



def load_image_test(image_file):
  input_image = load(image_file)
  input_image = resize(input_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, _ = normalize(input_image, input_image)

  return input_image





train_dataset = tf.data.Dataset.from_tensor_slices(
    [num for num in range(21, 41)])

train_dataset = train_dataset.map(load_image_train,
                                num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
