# Tuts from:
# https://towardsdatascience.com/keras-transfer-learning-for-beginners-6c9b8b7143e

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


# In[2]:


base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)

x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(3,activation='softmax')(x) #final layer with softmax activation


# In[3]:

# Using Model - Original Tuts
model=Model(inputs=base_model.input,outputs=preds) 

#specify the inputs
#specify the outputs
#now a model has been created based on our architecture


# In[4]:

# Original: not training the first 20 layers
for layer in model.layers[:50]:
    layer.trainable=False
for layer in model.layers[50:]:
    layer.trainable=True

# print(len(model.layers)) # 92 layers

# for layer in model.layers:
#     layer.trainable = False

# In[5]:


train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

train_generator=train_datagen.flow_from_directory('./train/', # this is where you specify the path to the main data folder
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)


# In[6]:


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

step_size_train=train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=5)

# In[7]:
# Actually using the model to predict/classify new images
import cv2
path = './validation/horse1.jpg'
pic = cv2.imread(path)
# print(img.shape) # test if image loaded correctly
pic1 = cv2.resize(pic, (224, 224))
# print(img.shape) # test image was resized

# Your input should be of shape: [1, image_width, image_height, number_of_channels = RGB:3 channels]
# img = np.reshape(img, [1,224,224,3]) # this option also work
img = np.expand_dims(pic1, axis=0) 
# print(img.shape)

classes = train_generator.class_indices
print(classes)

y_validation = model.predict(img, verbose=0)
y_classes = y_validation.argmax(axis=-1)

print(y_validation[0])
print(y_classes)

plt.imshow(pic1)
plt.show()