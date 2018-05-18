from keras.preprocessing.image import array_to_img, load_img
from skimage import feature
from skimage import exposure
import os
import numpy as np

crp_factor = 2
ds_factor = 3

projectFolder = 'D:/Data Science/NYU Data Science/DS-GA 1003 Project'
figures_dir_path = projectFolder + '/Report/Figures'

# Original
file_name = 'Grey_Cropped_Downsampled.jpg'
raw_img = load_img(os.path.join(figures_dir_path, file_name), grayscale=True)

# HOG: Histogram of Oriented Gradients
fd, hog_image = feature.hog(raw_img, visualise=True, feature_vector=True)
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
hog_img = array_to_img(hog_image_rescaled[:,:,np.newaxis])
hog_img.save(os.path.join(figures_dir_path,'HOG.jpg'))

# LBP: Local Binary Pattern
lbp_image = feature.local_binary_pattern(raw_img, 2, 20)
lbp_image_rescaled = exposure.rescale_intensity(lbp_image, in_range=(0, 10))
lbp_img = array_to_img(lbp_image_rescaled[:,:,np.newaxis])
lbp_img.save(os.path.join(figures_dir_path,'LBP.jpg'))

# DAISY
descs, descs_img = feature.daisy(raw_img, step=20, visualize=True, radius=10, rings=2, histograms=2)
daisy_img = array_to_img(descs_img)
daisy_img.save(os.path.join(figures_dir_path,'DAISY.jpg'))



