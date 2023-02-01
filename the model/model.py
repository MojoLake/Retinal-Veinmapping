import tensorflow as tf

import os
import pathlib
import time
import datetime

from matplotlib import pyplot as plt

from IPython import display

# own files and functions
from image_processing import load, create_more, load_image_test
from generator import Generator, generator_loss
from discriminator import Discriminator, discriminator_loss


# the path from which the training data is found
PATH = r"C:\Users\elias\ML\Retinal-Veinmapping\Dataset"

path = pathlib.Path(PATH)



BUFFER_SIZE = 20

BATCH_SIZE = 1

IMG_WIDTH = 256
IMG_HEIGHT = 256



real_list = [load(str(path / fr'training\images_256\{num}_train.jpg')) for num in range(21, 41)]
target_list = [load(str(path / fr'training\1st_manual_256\{num}_manual.jpg')) for num in range(21, 41)]

real_set = tf.data.Dataset.from_tensor_slices(real_list)
target_set = tf.data.Dataset.from_tensor_slices(target_list)
                            

train_dataset = tf.data.Dataset.zip((real_set, target_set))
train_dataset = train_dataset.map(create_more, 
                                 num_parallel_calls=tf.data.AUTOTUNE)

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)


test_list = [load_image_test(str(path / fr'test\jpg_image_256\{num}_test.jpg')) for num in range(1, 21)]

test_dataset = tf.data.Dataset.from_tensor_slices(test_list)
test_dataset = test_dataset.batch(BATCH_SIZE)



OUTPUT_CHANNELS = 3


LAMBDA = 100 # original 100


generator = Generator()

discriminator = Discriminator()



generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)




def generate_images(model, test_input):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))
  plt.imshow(prediction[0])
  plt.axis('off')
  plt.tight_layout()
  plt.show()

  display_list = [test_input[0], prediction[0]]
  title = ['Input Image', 'Predicted Image']

  for i in range(2):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()





log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))




@tf.function
def train_step(input_image, target, step):
  print("Train_step:", input_image.shape)
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target, LAMBDA)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
    tf.summary.scalar('disc_loss', disc_loss, step=step//1000)




  
def fit(train_ds, test_ds, steps):
  example_input = next(iter(test_ds.take(1)))
  example_input = example_input[:,:,:,:3] # reshape to get correct dimensions
  start = time.time()

  for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
    input_image = input_image[:,:,:,:3]
    target = target[:,:,:,:3]
    print(step)
    if (step) % 200 == 0:
      display.clear_output(wait=True)

      if step != 0:
        print(f'Time taken steps: {time.time()-start:.2f} sec\n')

      start = time.time()

      generate_images(generator, example_input)
      print(f"Step: {step//10}k")

    train_step(input_image, target, step)
    # inp[None,:,:,:]

    # Training step
    if (step+1) % 10 == 0:
      print('.', end='', flush=True)


    # Save (checkpoint) the model every 200 steps
    if (step + 1) % 200 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)


def generate_test_images():
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    
    for img in test_dataset.take(20):
        img = img[:,:,:,:3] # awkward reshaping...
        generate_images(generator, img)


# fit makes the magic happen
# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    
# fit(train_dataset, test_dataset, steps=40000)


# test the model on the test data unseen to the model
generate_test_images()






