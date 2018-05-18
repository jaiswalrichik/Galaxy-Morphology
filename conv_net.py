from keras.models import Sequential
from keras.layers.core import Lambda, Flatten, Activation, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
import numpy as np
from time import time

### ---- Convolutional Neural Network functions
def create_conv_pool(model, nFilters):
    '''
    create convolution / pooling block
    '''
    model.add(Conv2D(nFilters, kernel_size=(3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    
def create_fully_connected(model, nUnits, activation, dropout_bool=False):
    '''
    create fully connected block
    '''
    model.add(Dense(nUnits))
    model.add(Activation(activation))
    if dropout_bool:
        model.add(Dropout(0.5))

def CNN_simple(input_size=(71,71,3), output_size=5):
    '''
    Implements a convolutional neural network
    '''
    
    # create sequential model
    model = Sequential()

    # add input layer
    model.add(Lambda(lambda x : x, input_shape=input_size))
    
    # use a simple architecture
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', kernel_initializer='random_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    
    create_conv_pool(model, 32)
    create_conv_pool(model, 64)

    # convert 3D feature to 1D vector
    model.add(Flatten())
    
    # add fully connected layers
    create_fully_connected(model, 64, activation='relu', dropout_bool=True)
    create_fully_connected(model, output_size, activation='softmax')
    
    return model

def CNN_VGG16(input_size=(71,71,3), output_size=5):
    '''
    Implements a convolutional neural network
    '''
    
    # create sequential model
    model = Sequential()
    
    # use architecture of VGG-16
    model.add(Conv2D(64, (3, 3), input_shape=input_size, activation='relu', padding='same', kernel_initializer='random_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(output_size, activation='softmax'))
    
    return model

def save_bottleneck_features(train_generator, validation_generator, 
                             nb_train_samples, nb_validation_samples, 
                             batch_size, models_dir_path):
    
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    bottleneck_features_train = model.predict_generator(
        train_generator, nb_train_samples // batch_size)
    np.save(open(models_dir_path + 'bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)

    bottleneck_features_validation = model.predict_generator(
        validation_generator, nb_validation_samples // batch_size)
    np.save(open(models_dir_path + 'bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)


def train_top_model(train_labels, val_labels, batch_size, 
                    epochs, models_dir_path, output_size):
    
    train_data = np.load(open(models_dir_path + 'bottleneck_features_train.npy'))
    validation_data = np.load(open(models_dir_path + 'bottleneck_features_validation.npy'))

    model = Sequential()
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(output_size, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    t0 = time()
    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, val_labels))
    model.save_weights(models_dir_path + 'top_model_weights')
    print("done in %0.3fs" % (time() - t0))
    
    return model
