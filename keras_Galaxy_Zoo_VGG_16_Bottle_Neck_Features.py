import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from time import time

from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model

import data_utils
import conv_net

projectFolder = 'D:/Data Science/NYU Data Science/DS-GA 1003 Project'
data_dir_path = projectFolder + '/Data/'
models_dir_path = projectFolder + '/Saved Models/'

n_classes = 5
classNames = {0: 'Disc',
              1: 'Spiral',
              2: 'Elliptical',
              3: 'Round',
              4: 'Other'}

batch_size = 16

### Data handler
sample_fractions = [0.6, 0.3, 0.1] # training / validation / testing

if K.image_data_format() == 'channels_first':
    input_size = (3,71,71)    
else:
    input_size = (71,71,3)

output_size = 5

handler = data_utils.data_handler(data_dir_path, sample_fractions=sample_fractions, 
                              input_size=input_size, labels_type='classes', 
                              output_size=output_size, normalize_input=False, 
                              create_samples_bool=False, preprocess_bool=False, 
                              crp_factor=2, ds_factor=3)

### Load data
X_train, y_train = data_utils.load_samples(handler, 'training', grey_scale=False, as_vector=False)    
X_val, y_val = data_utils.load_samples(handler, 'validation', grey_scale=False, as_vector=False) 

### Convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, n_classes)
y_val = np_utils.to_categorical(y_val, n_classes)

nb_train_samples = len(y_train)
nb_validation_samples = len(y_val)

# this is the augmentation configuration for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=90,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# this is the augmentation configuration for validation
val_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures and generate batches of augmented images 
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
validation_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

### Save Bottleneck Features
conv_net.save_bottleneck_features(train_generator, validation_generator, 
                                  nb_train_samples, nb_validation_samples, 
                                  batch_size, models_dir_path)
epochs = 200
model = conv_net.train_top_model(y_train, y_val, batch_size, 
                    epochs, models_dir_path, output_size)
### Evaluation
y_pred = model.predict(X_val)

y_pred_class = np.argmax(y_pred,axis=1)
y_val_class = np.argmax(y_val,axis=1)

print(metrics.classification_report(y_val_class, y_pred_class, target_names=classNames.values()))

conf_matrix = metrics.confusion_matrix(y_val_class, y_pred_class, labels=range(n_classes))
print(conf_matrix)

sns.heatmap(conf_matrix, cmap='hot')
plt.title('Confusion Matrix of VGG16 Trained from Sratch', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.yticks(range(n_classes), classNames.values(), fontsize=10, rotation='horizontal')
plt.xticks(range(n_classes), classNames.values(), fontsize=10, rotation='vertical')
