# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#import os
#from os import listdir
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from tensorflow.keras.layers import Activation,BatchNormalization,GlobalAveragePooling2D,UpSampling2D
from tensorflow.keras.callbacks import History
#from tensorflow.keras.optimizers import Adam,SGD
#from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

train_datagen = ImageDataGenerator(
                 rescale=1./255, 
                 zoom_range = 0.2,
                 shear_range=0.2,
                horizontal_flip=True
                 )
test_datagen = ImageDataGenerator(rescale=1./255)

valid_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(r'D:\DATA\Train',
                                                target_size=(319,319),
                                                batch_size = 157,
                                                class_mode = 'categorical')

validation_set = valid_datagen.flow_from_directory(r'D:\DATA\Validation',
                                           target_size=(319,319),
                                           batch_size = 18,
                                           class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(r'D:\DATA\Test',
                                           target_size=(319,319),
                                           batch_size = 15,
                                           class_mode = 'categorical')
def plot_graph(history):
    
    figure = plt.figure()
    plt.subplot(2,1,1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train,valid'],loc = 'lower right')
    
    plt.subplot(2,1,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train','valid'],loc = 'upper right')
    
def run_model():    
    
    tf.random.set_seed(0)
    model = Sequential()
    model.add(Conv2D(96,kernel_size=(3,3),strides=(3,3),input_shape=(210,210,3),activation = 'relu'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (3,3),strides = (2,2)))
    model.add(Conv2D(64,kernel_size=(1,1)))
    model.add(Conv2D(64,kernel_size=(3,3)))
    model.add(Conv2D(256,kernel_size=(1,1)))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (3,3),strides = (2,2)))
    model.add(Conv2D(96,kernel_size=(1,1)))
    model.add(Conv2D(96,kernel_size=(3,3)))
    model.add(Conv2D(384,kernel_size=(1,1)))
    #model.add(BatchNormalization())
    model.add(Conv2D(64,kernel_size=(1,1)))
    model.add(Conv2D(64,kernel_size=(3,3)))
    model.add(Conv2D(256,kernel_size=(1,1)))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024,activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(512,activation = 'relu'))
    model.add(Dense(6,activation='softmax'))
    print(model.summary())
    #opt = SGD(lr = 0.01,momentum=0.9)
    #adam = Adam(lr = 1e-3)
    model.compile(loss = 'categorical_crossentropy',metrics = ['accuracy'],optimizer = 'Adam')
    history = History()
    #history = model.fit(trainX,trainY,validation_data=(validX,validY),batch_size = BATCH_SIZE,epochs = EPOCHS)
    history = model.fit_generator(training_set,steps_per_epoch = 2,epochs = 200,validation_data = test_set,validation_steps = 7)
    plot_graph(history) 
    #model.save('model_74-accuracy.h5')
    
def get_model():
    tf.random.set_seed(0)
    model = Sequential()
    model.add(Conv2D(96,kernel_size=(13,13),strides=(3,3),input_shape=(319,319,3),activation = 'relu'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (3,3),strides = (2,2)))
    model.add(Conv2D(256,kernel_size=(3,3),strides=(3,3)))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (3,3),strides = (2,2)))
    model.add(Conv2D(384,kernel_size=(1,1)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024,activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(512,activation = 'relu'))
    model.add(Dense(3,activation='softmax'))
    print(model.summary())
    #opt = SGD(lr = 0.01,momentum=0.9)
    #adam = Adam(lr = 1e-3)
    model.compile(loss = 'categorical_crossentropy',metrics = ['accuracy'],optimizer = 'Adam')
    history = History()
    #history = model.fit(trainX,trainY,validation_data=(validX,validY),batch_size = BATCH_SIZE,epochs = EPOCHS)
    history = model.fit_generator(training_set,steps_per_epoch = 3,epochs = 20,validation_data = validation_set,validation_steps = 6)
    plot_graph(history)
    print("Evaluating on test set")
    results = model.predict_generator(test_set,verbose=1,steps=5)
    print("test loss, test acc:", results)
    model.save('my_model.h5')
    return model
#run_model() 

#def predict_classes(full_filename):
   # full_filename = ''  #image_path
   # img=image.load_img(full_filename,target_size=(319,319),color_mode='rgb')
   # img=image.img_to_array(img)
   # img=img/255
   # img=np.expand_dims(img,axis=0)
   # prediction=str(mymodel.predict_classes(img)[0])
   # return prediction
   
mymodel = get_model()
mymodel.save('my_model.h5')