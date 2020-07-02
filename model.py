import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Input
from keras import regularizers
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout

class model_classification:
    def _init_(self,input = 512):
        input_image = Input(shape=[512,512,3],name="input_image")

        resnet = ResNet50(input_tensor=input_image,weights='imagenet',include_top=False,pooling=None)
        model = Sequential()
        model.add(resnet)
        model.add(Flatten(input_shape=(16,16,2048)))
        model.add(Dense(64,activation='relu'))
        model.add(Dense(1,activation='sigmoid'))
        
        self.model = model
        self.input_image = input_image