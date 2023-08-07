# %%
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
#loading images
import io
from PIL import Image
import os
#plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#classification report
from sklearn.metrics import classification_report

# Path to the directory containing the images (both bleached and healthy)
# %%
data_dir = "../data/"

# %%
# path to the image directories
normal_directory = '../data/normal/'
covid19_directory = '../data/covid19/'


# %%
# create dataframes
normal_df = create_df("../data/normal")
covid19_df = create_df("../data/covid19")


# %%
# Image dimensions, batch size, and train/test split ratio
img_width, img_height = 224, 224
batch_size = 32
validation_split = 0.2

#Creating data generator with data augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=validation_split
)

#Splitting the data into train and test sets
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

test_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# %%
# Loading the pre-trained VGG16 model 
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3)) #not including fully connected top layes

for layer in vgg_model.layers:
    layer.trainable = False

# Creating a new model and adding the pre-trained VGG16 model as a layer
model = Sequential()
model.add(vgg_model)

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#Compiling
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Number of epochs to train the model
epochs = 10

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size
)

# %%
# Evaluating the model on the test set
_, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# %%
predictions = model.predict(train_generator, batch_size=128)

# %%
test_steps = len(test_generator)
y_pred = model.predict(test_generator, steps=test_steps)
y_pred = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
# %%
# Converting true labels from generator to array
test_generator.reset()
y_true = []
for i in range(test_steps):
    _, labels = test_generator.next()
    y_true.extend(labels)
y_true = np.array(y_true)

y_true
# %%
#labels to 1-dimensional arrays
y_pred = y_pred.reshape(-1)
y_true = y_true.reshape(-1)

# Generating classification report
print('Generating classification report.. find results in "out" folder')
target_names = ['normal', 'covid19']
report = classification_report(y_true, y_pred, target_names=target_names)

print(report)

#Saving classification report as text file
report_path = "../out/classification_report.txt"

text_file = open(report_path, "w")
n = text_file.write(report)
text_file.close()

# %%
#Histogram of model fitness
def plot_learning_curve(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, label='training acc')
    plt.plot(epochs, val_acc, label='validation acc')
    plt.legend();
    plt.figure();

    plt.plot(epochs, loss, label='training loss')
    plt.plot(epochs, val_loss, label='validation loss')
    plt.legend();
    plt.savefig("../out/loss_acc_curve.png", format="png") # specify filetype explicitly

plot_learning_curve(history)