import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
import tensorflow_io as tfio



r_path = r"C:\Users\robin\OneDrive\Työpöytä\Retinas\Retinal-Veinmapping\Dataset\training\images\21_training.tif"
e_path = r"C:\Users\elias\OneDrive\Työpöytä\code\ml\Retinal-Veinmapping\Dataset\training\images\21_training.tif"


def parse_image(img_path: str) -> dict:
    image = tf.io.read_file(img_path)
    tfio.experimental.image.decode_tiff(image)


parse_image(e_path)
