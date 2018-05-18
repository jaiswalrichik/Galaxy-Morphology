from keras.preprocessing.image import load_img
import data_utils
import matplotlib.pyplot as plt
import os
from collections import Counter

projectFolder = 'D:/Data Science/NYU Data Science/DS-GA 1003 Project'
data_dir_path = projectFolder + '/Data'
training_images = 'D:/Data Science/NYU Data Science/DS-GA 1003 Project/Data/training_data/Preprocessed'
figures_dir_path = projectFolder + '/Report/Figures'

classNames = {0: 'Disc',
              1: 'Spiral',
              2: 'Elliptical',
              3: 'Round',
              4: 'Other'}

### --- Distriution of Classes
sample_fractions = [0.6, 0.3, 0.1] # training / validation / testing
input_size = (71,71)
output_size = 1

handler = data_utils.data_handler(data_dir_path, sample_fractions=sample_fractions, 
                              input_size=input_size, labels_type='classes', 
                              output_size=output_size, normalize_input=False, 
                              create_samples_bool=False, preprocess_bool=False, 
                              crp_factor=2, ds_factor=3)

labels = handler.labels
count = Counter(labels.values())
classes_count = dict((classNames[key], value) for (key, value) in count.items())

plt.bar(classes_count.keys(), classes_count.values())
plt.title('Class Distribution')
plt.savefig(os.path.join(figures_dir_path, 'Training Classes Histogram.png'))

### --- Example from Each Class

file_name = '147244.jpg'
className = classNames[handler.find_label(handler.get_image_id(file_name))]
raw_img = load_img(os.path.join(training_images, file_name))
raw_img.save(os.path.join(figures_dir_path,className + '.jpg'))

file_name = '174019.jpg'
className = classNames[handler.find_label(handler.get_image_id(file_name))]
raw_img = load_img(os.path.join(training_images, file_name))
raw_img.save(os.path.join(figures_dir_path,className + '.jpg'))

file_name = '108305.jpg'
className = classNames[handler.find_label(handler.get_image_id(file_name))]
raw_img = load_img(os.path.join(training_images, file_name))
raw_img.save(os.path.join(figures_dir_path,className + '.jpg'))

file_name = '107975.jpg'
className = classNames[handler.find_label(handler.get_image_id(file_name))]
raw_img = load_img(os.path.join(training_images, file_name))
raw_img.save(os.path.join(figures_dir_path,className + '.jpg'))

file_name = '142581.jpg'
className = classNames[handler.find_label(handler.get_image_id(file_name))]
raw_img = load_img(os.path.join(training_images, file_name))
raw_img.save(os.path.join(figures_dir_path,className + '.jpg'))
