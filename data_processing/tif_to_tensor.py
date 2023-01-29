import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
import tensorflow_io as tfio
from pathlib import Path


r_path = r"C:\Users\robin\OneDrive\Työpöytä\Retinas\Retinal-Veinmapping\Dataset\training\gifimages"
e_path = "C:\\Users\\elias\\ML\\Retinal-veinmapping\\dataset\\training\\1st_manual"

r_path2= r"C:\Users\robin\OneDrive\Työpöytä\Retinas\Retinal-Veinmapping\Dataset\training\gifimages\21_training.gif"

# data_dir = pathlib.Path(r_path).with_suffix('')

def parse_image(img_path: str) -> dict:
    image = tf.io.read_file(img_path)
    ImageTensor = tf.io.decode_gif(image)
    return ImageTensor


# print(parse_image(r_path2))


image_count = len(list(Path(e_path).glob('*')))
print(image_count)



batch_size = 5
img_height = 584
img_width = 565

train_ds = tf.keras.utils.image_dataset_from_directory(
  directory=e_path,
  validation_split=0.1,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)