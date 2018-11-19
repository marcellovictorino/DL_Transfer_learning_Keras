# Tuts from:
# https://towardsdatascience.com/keras-transfer-learning-for-beginners-6c9b8b7143e
# Additional reference: https://towardsdatascience.com/how-to-train-your-model-dramatically-faster-9ad063f0f718

# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.optimizers import Adam
import cv2

# In[2]:

base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

base_input = base_model.get_layer(index=0).input
base_output = base_model.get_layer(index=-2).output

base_output = GlobalAveragePooling2D()(base_output) # Important to set the expected shape = [?, height, width, 3 channels]

bottleneck_model = Model(inputs=base_input, outputs=base_output)
print(base_output) # the model has learned a 1024 dimensional representation of any image input
#specify the inputs
#specify the outputs
#now a model has been created based on our architecture

# In[4]:

# Freeze all already traiined layers = not training them
for layer in bottleneck_model.layers:
    layer.trainable = False

# In[5]:
# Create the Sequential model

new_model = Sequential()
new_model.add(bottleneck_model)
new_model.add(Dense(3, activation='softmax', input_dim=1024)) 
# 3 classes
# softmax activation to ensure image class outputs can be interpreted as probabilities


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

train_generator = train_datagen.flow_from_directory('./train/', # this is where you specify the path to the main data folder
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

# In[6]:


new_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

step_size_train = train_generator.n // train_generator.batch_size

new_model.fit_generator(generator=train_generator, steps_per_epoch=step_size_train, epochs=5)

# In[7]:
# Actually using the model to predict/classify new images

path = './validation/dog2.jpg'
pic = cv2.imread(path)
pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
# print(img.shape) # test if image loaded correctly
pic1 = cv2.resize(pic, (224, 224))
# print(img.shape) # test image was resized

# Your input should be of shape: [1, image_width, image_height, number_of_channels = RGB:3 channels]
# img = np.reshape(img, [1,224,224,3]) # this option also work
img = np.expand_dims(pic1, axis=0) 
# print(img.shape)

classes = train_generator.class_indices
print(classes)

y_validation = new_model.predict(img, verbose=0)
y_classes = y_validation.argmax(axis=-1)

# prediction = new_model.predict_classes(img, verbose=0) # Only available for Sequential model

print(y_validation[0])
print(y_classes)

# print(prediction)
plt.imshow(pic1)
plt.show()

# Final result: model absurdly overfitted?
# Only classify images as CATS!