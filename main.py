from __future__ import print_function, division

''' DESIGN DOC

TASK: sythesis of fake retinal images, cisir summer 2019


'''

from PIL import Image
import os
import numpy as np
import random
import glob
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.layers import Reshape, Input, Dropout
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import SGD, Adam
import argparse
import math
import matplotlib.pyplot as plt

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
  image = tf.image.resize(image, [64, 64])
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
BATCH_SIZE=10
ds  = ds.repeat()
ds  = ds.batch(BATCH_SIZE)
ds  = ds.prefetch(buffer_size=AUTOTUNE)
keras_ds = ds.map(change_range)

## testing the dataset object output
for i in range(10):
    image = next(iter(keras_ds))
    print(i, image.shape)
    print(np.min(image), np.max(image))
    image = image*0.5 + 0.5
    plt.imshow(image[0,:,:,:])
    plt.axis('off')
    plt.savefig("./images/processed_%d.png" % i)

def conv2d(filters, kernel_size, strides, padding):
    return Conv2D(filters=filters, kernel_size=kernel_size, 
                    strides=strides, padding=padding, 
                    kernel_initializer='he_normal')

def conv2dTranspose(filters, kernel_size, strides, padding):
    return Conv2D(filters=filters, kernel_size=kernel_size,
                    strides=strides, padding=padding, 
                    kernel_initializer='he_normal')
#---------------------------------------


class DCGAN():
    def __init__(self):
        self.height = 64
        self.width  = 64
        self.channel= 3
        self.img_shape=(self.width, self.height, self.channel)
        self.latent_dim = 100
        layer_init = 'he_normal'  #kaiming he initializer

        optimizer = Adam(0.0002, 0.5)
        #optimizer = SGD(lr=0.00005)
        # build models
        self.dis = self._build_discriminator()
        # compile discriminator,  freeze discriminator pipe g to d, 
        self.dis.compile(loss='binary_crossentropy', 
                optimizer=optimizer,
                metrics=['accuracy'])

        # build generator
        self.gen = self._build_generator()
        z = Input(shape=(self.latent_dim,))
        img = self.gen(z)
        self.dis.trainable = False
        valid = self.dis(img)

        # combined model
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', 
                    optimizer=optimizer)


    def _build_discriminator(self, padding='same'):
        
        model = tf.keras.Sequential()
        model.add(Conv2D(64, (3,3), strides=(2,2),
                input_shape=(self.height, self.width, self.channel),
                kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, (3,3), strides=(2,2), padding=padding,
            kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Conv2D(16, (3,3), strides=(2,2), padding=padding,
            kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        
        model.summary()
        x = Input(shape=self.img_shape)
        prob = model(x)
        return Model(x, prob)


    def _build_generator(self, padding='same'):

        model = tf.keras.Sequential()
        model.add(Dense(16*16*64, input_shape=(self.latent_dim,)))
        #model.add(Activation('relu'))
        model.add(Activation('relu'))
        model.add(Reshape((16, 16, 64),  input_shape=(16*16*64,)))
        model.add(Conv2DTranspose(64, kernel_size=(3,3), strides=(2,2),
                padding=padding, kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(Conv2DTranspose(64, kernel_size=(3,3), strides=(1,1),
                padding=padding, kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(Conv2DTranspose(32, 
                kernel_size=(3,3), strides=(2,2),
                padding=padding, kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(Conv2DTranspose(self.channel,
                kernel_size=(3,3), strides=(1,1),
                padding=padding, kernel_initializer='he_normal'))
        model.add(Activation('tanh'))
            
        model.summary()
        noise = Input(shape=(self.latent_dim,))
        img   = model(noise)

        return Model(noise, img)
               
    def train(self, epochs=10000, batch_size=32, save_interval=1000):
        
        # adversarial ground trutg
        valid = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))
        
        for epoch in range(epochs+1):

            imgs = next(iter(keras_ds))
            #--------------------------
            # train discriminator
            #----------------------------
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.gen.predict(noise)

            d_loss_real = self.dis.train_on_batch(imgs, valid)
            d_loss_fake = self.dis.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            #----------------------------
            #  train generator
            #-----------------------------
            g_loss = self.combined.train_on_batch(noise, valid)
            print("%4d [D loss: %4.3f, acc.: %.2f%%] [g loss: %4.3f]" %
                    (epoch, d_loss[0], d_loss[1]*100, g_loss))
            
            if epoch % save_interval == 0:
                print("== saving images, ", epoch)
                self.save_imgs(epoch)


    def save_imgs(self, epoch):
        r, c = 5,5
        noise = np.random.normal(0, 1, (r*c, self.latent_dim))
        gen_imgs=  self.gen.predict(noise)

        # rescale images 0-1
        gen_imgs = 0.5*gen_imgs + 0.5

        fig, axs = plt.subplots(r,c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("./images/mnist_%d.png" % epoch)
        plt.close()

gan = DCGAN()
gan.train(batch_size=BATCH_SIZE)
'''mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print(x_train.shape)
print(type(x_train))
'''
