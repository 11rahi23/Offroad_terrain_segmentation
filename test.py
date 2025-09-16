import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import tensorflow as tf
from tensorflow import keras
#import keras.api._v2.keras as keras
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image
import segmentation_models as sm
from keras.metrics import MeanIoU
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.utils import to_categorical, load_img
from sklearn.model_selection import train_test_split
from keras.models import load_model
from tqdm import tqdm
import random
import pickle
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.optimizers import SGD, Adam 

mask_labels=pd.read_csv("E:/Offroad database/Rest of the files/class_dict_no_non_traversable_low_vegetation.csv")

model = load_model('segmented_yamaha_augmented_8_class_sampled_data_512_3_resnet34.h5',compile=False)

video_path='E:/Offroad database/Rest of the files/Test Videos/Ascience_4vehicle_LW_Run01.mp4'
cap=cv2.VideoCapture(video_path)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
plt.ion()

colormap = {}
for index, row in mask_labels.iterrows():
    class_index = index  # Class index from the DataFrame
    rgb_color = [row.iloc[1], row.iloc[2], row.iloc[3]]  # RGB color values from the DataFrame
    colormap[class_index] = rgb_color

frame_counter=0
while True:
    ret, validation_img1=cap.read()
    if not ret:
        print("End of video")
        break

    if frame_counter % 10 == 0:
        validation_img=cv2.cvtColor(validation_img1, cv2.COLOR_BGR2RGB)
        validation_img=cv2.resize(validation_img, (768, 768))
        validation_img=np.array(validation_img)
        validation_img=np.expand_dims(validation_img, 0)
        prediction=model.predict(validation_img)
        predicted_img=np.argmax(prediction, axis=3)[0,:,:]
    
        predicted_color_img = np.zeros((predicted_img.shape[0], predicted_img.shape[1], 3), dtype=np.uint8)    

        for class_index, rgb_color in colormap.items():
            predicted_color_img[predicted_img == class_index] = rgb_color
    
        ax1.clear()
        ax1.set_title('Video feed')
        ax1.imshow(cv2.cvtColor(validation_img1, cv2.COLOR_BGR2RGB))
        ax1.axis('off')

        # Display the label video in the second subplot
        ax2.clear()
        ax2.set_title('Prediction Label')
        ax2.imshow(predicted_color_img)
        ax2.axis('off')

        # Update the figure to show the new frames
        plt.draw()
        plt.pause(0.01)
    
    frame_counter += 1

    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break


cap.release()
cv2.destroyAllWindows()

plt.ioff()
plt.show()