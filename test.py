from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from flask import Flask, redirect, url_for, request, render_template, app
from werkzeug.utils import secure_filename
import os

model = load_model('monkey_breed_mobilNet.h5')

#img_rows, img_cols = 224, 224

# validation_data_dir = '/Users/durgeshthakur/Deep Learning Stuff/Monkey Breed Classifier/monkey_breed/validation'

# validation_datagen = ImageDataGenerator(rescale=1./255)

# batch_size = 32

# validation_generator = validation_datagen.flow_from_directory(
#                             validation_data_dir,
#                             target_size=(img_rows,img_cols),
#                             batch_size=batch_size,
#                             class_mode='categorical')

# class_labels = validation_generator.class_indices
# class_labels = {v: k for k, v in class_labels.items()}
# classes = list(class_labels.values())
# print(classes)


class_labels = [
    'mantled_howler',
    'patas_monkey',
    'bald_uakari',
    'japanese_macaque',
    'pygmy_marmoset',
    'white_headed_capuchin',
    'silvery_marmoset',
    'common_squirrel_monkey',
    'black_headed_night_monkey',
    'nilgiri_langur'
]


def check(path):
    # prediction
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x.astype('float32') / 255
    pred = np.argmax(model.predict(x))

    print("It's a {}.".format(class_labels[pred]))
check(r'C:\Users\ASUS\Downloads\download.jpg')
