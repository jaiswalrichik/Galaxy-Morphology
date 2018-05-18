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

### Build model 
model = conv_net.CNN_VGG16(input_size=input_size, output_size=n_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

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

### Train model
t0 = time()
history = model.fit_generator(
        train_generator,
        steps_per_epoch=y_train.shape[0]// batch_size,
        epochs=200,
        validation_data=validation_generator,
        validation_steps=y_val.shape[0] // batch_size)
print("done in %0.3fs" % (time() - t0))

score = model.evaluate(X_train, y_train)
print("Training accuracy:", score[1])

score = model.evaluate(X_val, y_val)
print("Validation accuracy:", score[1])

plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
#plt.legend(['Training','Validation'])

### Save weights
model.save_weights(models_dir_path + '/VGG16_from_sratch.h5')

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
