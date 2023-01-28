import tensorflow as tf
import tensorflow_io as tfio

path = r"C:\Users\robin\OneDrive\Työpöytä\Retinas\Retinal-Veinmapping\Dataset\training\images\21_training.tif"


def parse_image(img_path: str) -> dict:
    image = tf.io.read_file(img_path)
    tfio.experimental.image.decode_tiff(image)


parse_image(path)
