import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

# Set the paths to the folders containing the images
with_mask_folder = 'D:/mask/new_custom/train/1'
without_mask_folder = "D:/mask/new_custom/train/0"

# Function to load and resize images
def load_and_resize_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize images to 128x128
            images.append(img)
    return images

# Load and resize images with masks
with_mask_images = load_and_resize_images(with_mask_folder)

# Load and resize images without masks
without_mask_images = load_and_resize_images(without_mask_folder)

# Create labels for the images
with_mask_labels = np.ones(len(with_mask_images))  # Label 1 for images with masks
without_mask_labels = np.zeros(len(without_mask_images))  # Label 0 for images without masks

# Concatenate the images and labels
X = np.array(with_mask_images + without_mask_images)
Y = np.concatenate((with_mask_labels, without_mask_labels), axis=0)

# **Train Test Split**

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Print shapes for confirmation
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

# scaling the data

X_train_scaled = X_train/255

X_test_scaled = X_test/255


# **Building a Convolutional Neural Networks (CNN)**

num_of_classes = 2

model = keras.Sequential()

model.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(128,128,3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))


model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))


model.add(keras.layers.Dense(num_of_classes, activation='sigmoid'))

# compile the neural network
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

# training the neural network
history = model.fit(X_train_scaled, Y_train, validation_split=0.1, epochs=5)
model_save_path = 'D:/mask/model.h5'  # Replace with your desired filename and path
model.save(model_save_path)

# **Model Evaluation**

loss, accuracy = model.evaluate(X_test_scaled, Y_test)
print('Test Accuracy =', accuracy)

h = history

# plot the loss value
plt.plot(h.history['loss'], label='train loss')
plt.plot(h.history['val_loss'], label='validation loss')
plt.legend()
plt.show()

# plot the accuracy value
plt.plot(h.history['acc'], label='train accuracy')
plt.plot(h.history['val_acc'], label='validation accuracy')
plt.legend()
plt.show()
