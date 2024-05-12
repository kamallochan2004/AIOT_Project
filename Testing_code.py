import cv2
import numpy as np
from tensorflow.keras.models import load_model  # Assuming Keras model



model_path = "Enter the path of your model" #Enter the path of your trained model

model = load_model(model_path)
input_image_path =('Enter the Path of the image to be predicted:') #Enter the path of image to you want to test your model
input_image = cv2.imread(input_image_path)
input_image_resized = cv2.resize(input_image, (128,128))
input_image_scaled = input_image_resized/255
input_image_reshaped = np.reshape(input_image_scaled, [1,128,128,3])
input_prediction = model.predict(input_image_reshaped)
input_pred_label = np.argmax(input_prediction)

if input_pred_label == 1:
    print('The person in the image is wearing a mask')
else:
    print('The person in the image is not wearing a mask')
