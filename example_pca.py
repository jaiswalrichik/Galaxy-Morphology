import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import data_utils

projectFolder = 'D:/Data Science/NYU Data Science/DS-GA 1003 Project'
data_dir_path = projectFolder + '/Data'
figures_dir_path = projectFolder + '/Report/Figures'

# Load Data
sample_fractions = [0.6, 0.3, 0.1] # training / validation / testing
input_size = (71,71)
output_size = 1

handler = data_utils.data_handler(data_dir_path, sample_fractions=sample_fractions, 
                              input_size=input_size, labels_type='classes', 
                              output_size=output_size, normalize_input=False, 
                              create_samples_bool=False, preprocess_bool=False, 
                              crp_factor=2, ds_factor=3)

X_train, y_train = data_utils.load_samples(handler, sample='training', grey_scale=True)

# PCA Variance Explained
X_train_pca, pca = data_utils.compute_pca(X_train, n_comp=X_train.shape[1])
    
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance');
plt.title('Cumulative Variance Explained ')
plt.show()
plt.savefig(os.path.join(figures_dir_path, 'PCA_Variance_Explained.png'))

# Reconstruted Images
raw_img = X_train[5000,:].reshape(input_size)
vector_img = raw_img.reshape(np.prod(raw_img.shape))

plt.figure(figsize=(10,8))
plt.subplot(2, 3, 1)
io.imshow(raw_img, cmap="gray")
plt.title('Original Image')

for i, n_comp in enumerate([100,200,300,400,500]):
    X_train_pca, pca = data_utils.compute_pca(X_train, n_comp=n_comp)
    restored_vector_img = pca.inverse_transform(X_train_pca[5000,:])
    restored_img = restored_vector_img.reshape(raw_img.shape)
    plt.subplot(2, 3, i+2)
    plt.imshow(restored_img, cmap="gray")
    plt.title('%s PCA Components' %n_comp)

plt.tight_layout()
plt.show()
plt.savefig(os.path.join(figures_dir_path, 'PCA Restored Images.png'))  
