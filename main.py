from __future__ import print_function, division

from PIL import Image
import os
import random
import glob
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import SGD
import argparse
import math

print(tf.__version__)

AUTOTUNE = tf.data.experimental.AUTOTUNE

data_dir = '/tf/data'
data = os.path.join(data_dir, "*/images/*.jpg")
data_list = glob.glob(data)
print(type(data_list[0]))
#data_list = [ str(path) for path in data_list]
#print(type(data_list[0]))
print(data_list[:10])

#print(len(data_list))
assert len(data_list) == 40
random.shuffle(data_list)

# function ------------------------------------------

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [192, 192])
  image /= 255.0  # normalize to [0,1] range

  return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


def change_range(image):
    # convert from [0,1] to [-1,1]
    return 2*image-1

#==================================================


path_ds = tf.data.Dataset.from_tensor_slices(data_list)
#print('shape: ', repr(path_ds.output_shapes))
#print('type: ', path_ds.output_types)
print(path_ds)

ds = path_ds.map(load_and_preprocess_image)
BATCH_SIZE=32
ds  = ds.repeat()
ds  = ds.batch(BATCH_SIZE)
ds  = ds.prefetch(buffer_size=AUTOTUNE)



class DCGAN():
    def __init__(self):
        self.height = 192
        self.width  = 192
        self.channel= 3
        self.gen = self._build_generator()
        self.gen.summary()

        # discriminator
        print("== building discriminator ...")
        self.dis = self._build_discriminator()
        self.dis.summary()

    def _build_discriminator(self, padding='same'):
        
        model = tf.keras.Sequentail()
        model.add(Conv2D(64, (3,3), strides=(2,2),
                input_shape=(self.height, self.width, self.dim)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3,3), strides=(2,2), padding=padding))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(16, (3,3), strides=(2,2), padding=padding))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Activation('tanh'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        return model


    def _build_generator(self, padding='same'):

        model = tf.keras.Sequential()
        model.add(Dense(1024, input_shape=(100,)))
        model.add(Activation('relu'))
        model.add(Dense(128*32*32))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Reshape((32, 32, 128),  input_shape=(128*32*32,)))
        model.add(Conv2DTranspose(64, kernel_size=(3,3), strides=(2,2),
                padding=padding))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2DTranspose(43, kernel_size=(3,3), strides=(3,3),
                padding=padding))
        model.add(Activation('tanh'))
        return model
        
            

gan = DCGAN()
'''mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print(x_train.shape)
print(type(x_train))
'''
