from keras.preprocessing.image import img_to_array, array_to_img, load_img
import os
import data_utils

crp_factor = 2
ds_factor = 3

projectFolder = 'D:/Data Science/NYU Data Science/DS-GA 1003 Project'
figures_dir_path = projectFolder + '/Report/Figures'

# Original
file_name = 'Original.jpg'
raw_img = load_img(os.path.join(figures_dir_path, file_name))
x = img_to_array(raw_img)

# Cropping
x = img_to_array(raw_img)
x = data_utils.crop_image(x, factor=crp_factor)
new_img = array_to_img(x)
new_img.save(os.path.join(figures_dir_path,'Cropped.jpg'))

# Downsampling
x = data_utils.downsample_image(x, factor=ds_factor)
new_img = array_to_img(x)
new_img.save(os.path.join(figures_dir_path,'Cropped_Downsampled.jpg'))

# Gray
gray_img = load_img(os.path.join(figures_dir_path, file_name), grayscale=True)
x = img_to_array(gray_img)
x = data_utils.crop_image(x, factor=crp_factor)
x = data_utils.downsample_image(x, factor=ds_factor)
new_img = array_to_img(x)
new_img.save(os.path.join(figures_dir_path,'Grey_Cropped_Downsampled.jpg'))

