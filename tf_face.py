import tensorflow as tf
import os
import numpy as np

import matplotlib.pyplot as plt


train_dir = "/Users/SPA/PycharmProjects/Young/archive/train"
val_dir = "/Users/SPA/PycharmProjects/Young/archive/val"

IMAGE_SIZE = 224
BATCH_SIZE = 7

data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./255,
)
# val_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
#     rescale = 1./255,
#     validation_split=1
#
# )

train_generator = data_generator.flow_from_directory(
    train_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='training'
)

val_generator = data_generator.flow_from_directory(
    val_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset='training'
)

# print(train_generator)



for image_batch, label_batch in train_generator:
    # print("image batch: ", image_batch)
    # print("label_batch: ", label_batch)
    break

print(train_generator.class_indices)

labels = '\n'.join(sorted(train_generator.class_indices.keys()))

# print(labels)
# how to write on a text file
with open('labels.txt', 'w') as f:
    f.write(labels)

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

base_model.trainable = False

