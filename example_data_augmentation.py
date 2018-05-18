from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
import os
import numpy as np

projectFolder = 'D:/Data Science/NYU Data Science/DS-GA 1003 Project'
figures_dir_path = projectFolder + '/Report/Figures'

# Load Data
img = load_img(figures_dir_path+'/Cropped.jpg')

# Data Augmentation 
datagen = ImageDataGenerator(
        rotation_range=90,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

x = img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir=figures_dir_path+'/Data Augmentation', 
                          save_prefix='augmented', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely
        
# Main Figure
source_folder = os.path.join(figures_dir_path, 'Data Augmentation')
all_images = os.listdir(source_folder)

plt.figure(figsize=(10,8))
plt.subplot(2, 3, 1)
plt.imshow(img)
plt.title('Original Image')

i = 0
for i in np.arange(0,5):
    img = load_img(os.path.join(figures_dir_path, 'Data Augmentation', all_images[i]))
    plt.subplot(2, 3, i+2)
    plt.imshow(img)
    plt.title('Aumented Image #%s ' %(i+1))

plt.tight_layout()
plt.savefig(os.path.join(figures_dir_path, 'Augmented Images.jpeg'))  

