import numpy as np
import matplotlib.pylab as plt
import os
import csv
import shutil

from sklearn.decomposition import PCA
from keras.preprocessing.image import img_to_array, array_to_img, load_img
from skimage import io

np.random.seed(42)

### ---- Data Preparation Class

class data_handler:    
    """
    A class to create and handle training, validation and testing samples
    This class has methods for pre-processing and data augmentation
    """
    def __init__(self, data_path, sample_fractions=[0.6, 0.3, 0.1], 
                 input_size=(3,424,424), labels_type='classes', output_size=1, 
                 normalize_input=False, create_samples_bool=False, 
                 preprocess_bool=True, crp_factor=2, ds_factor=3):    
        
        assert np.abs(np.sum(sample_fractions) - 1) < 1e-6, "Sample fractions do not add up to 1 !!"
        self.sample_fractions = sample_fractions # training, validation and testing
        self.input_size = input_size
        self.labels_type = labels_type
        self.output_size = output_size
        
        self.data_path = data_path
        self.sample_paths = {'training': os.path.join(data_path, "training_data"),
                      'validation': os.path.join(data_path, "validation_data"),
                      'testing': os.path.join(data_path, "testing_data")}
        
        def create_samples(self):
            
            source_folder = os.path.join(self.data_path, 'images_training')
            
            all_images = os.listdir(source_folder)
            all_images = np.random.permutation(all_images)
            
            n_images = len(all_images)
            n_training = int(n_images * self.sample_fractions[0])
            n_validation = int(n_images * self.sample_fractions[1])
            
            print('Creating Training Sample')
            for img in all_images[0:n_training]:
                shutil.copy(os.path.join(source_folder, img), 
                            os.path.join(self.sample_paths['training'], 'Raw'))
                
            print('Creating Validation Sample')
            for img in all_images[n_training:n_training+n_validation]:
                shutil.copy(os.path.join(source_folder, img), 
                            os.path.join(self.sample_paths['validation'], 'Raw'))
            
            print('Creating Testing Sample')
            for img in all_images[n_training+n_validation:]:
                shutil.copy(os.path.join(source_folder, img), 
                            os.path.join(self.sample_paths['testing'],'Raw'))
            
            return self
        
        def preprocess_images(self, sample_type, crp_factor, ds_factor):
            
            source_folder = os.path.join(self.sample_paths[sample_type],'Raw')
            destination_folder = os.path.join(self.sample_paths[sample_type],'Preprocessed')
            
            raw_images = os.listdir(source_folder)
            
            for img_name in raw_images:
                try:
                    
                    raw_img = load_img(os.path.join(source_folder, img_name))
                    x = img_to_array(raw_img)
                    x = crop_image(x, factor=crp_factor)
                    x = downsample_image(x, factor=ds_factor)
                    
                    new_img = array_to_img(x)
                    new_img.save(os.path.join(destination_folder, img_name))
                    
                except:
                    print('Unable to load ' + img_name)
                
            return self
        
        if create_samples_bool:
            create_samples(self)
            
        if preprocess_bool:
            preprocess_images(self, 'training', crp_factor, ds_factor)
            preprocess_images(self, 'validation', crp_factor, ds_factor)
            preprocess_images(self, 'testing', crp_factor, ds_factor)
        
        def get_labels(self):
            
            labels_path = os.path.join(self.data_path, 'training_solutions.csv')
            
            if labels_type == 'probabilities':
                labels = proba_labels(labels_path)
            elif labels_type == 'classes':
                labels = classe_labels(labels_path)
            
            return labels
        
        self.labels = get_labels(self)
        
        def get_image_paths(sample_path):
            return [file for file in os.listdir(sample_path)]
        
        if preprocess_bool:
            self.training_images = get_image_paths(os.path.join(self.sample_paths['training'],'Preprocessed'))
            self.validation_images = get_image_paths(os.path.join(self.sample_paths['validation'],'Preprocessed'))
            self.testing_images = get_image_paths(os.path.join(self.sample_paths['testing'],'Preprocessed'))
        else:
            self.training_images = get_image_paths(os.path.join(self.sample_paths['training'],'Raw'))
            self.validation_images = get_image_paths(os.path.join(self.sample_paths['validation'],'Raw'))
            self.testing_images = get_image_paths(os.path.join(self.sample_paths['testing'],'Raw'))
        
    def get_image_id(self, fname):
        return fname.replace(".jpg","").replace("data","")
        
    def find_label(self, val):
        return self.labels[val]
    
def crop_image(img, factor):
    size_x = img.shape[0]
    size_y = img.shape[1]

    cropped_size_x = img.shape[0] // factor
    cropped_size_y = img.shape[1] // factor

    shift_x = (size_x - cropped_size_x) // 2
    shift_y = (size_y - cropped_size_y) // 2
    
    return img[shift_x:shift_x+cropped_size_x, shift_y:shift_y+cropped_size_y]
       
def downsample_image(img, factor):
    return img[::factor, ::factor]

def proba_labels(labels_path):
    
    labels = {}
    with open(labels_path, 'r') as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)
        for i, line in enumerate(reader):
            labels[line[0]] = [float(x) for x in line[1:]]
    
    return labels

def classe_labels(labels_path):
    '''
    disc = 0
    spiral = 1
    elliptical = 2
    round = 3
    other = 4
    '''
    labels = {}
    with open(labels_path, 'r') as f:  
        reader = csv.reader(f, delimiter=",")
        next(reader)
        for i, line in enumerate(reader):
            probas = [float(x) for x in line[1:]]
            
            a1 = probas[0:3] # Answers 1.1 - 1.3
            a2 = probas[3:5] # Answers 2.1 - 2.2
            #a3 = probas[5:7] # Answers 3.1 - 3.2
            a4 = probas[7:9] # Answers 4.1 - 4.2
            #a5 = probas[9:13] # Answers 5.1 - 5.4
            a6 = probas[13:15] # Answers 6.1 - 6.2
            a7 = probas[15:18] # Answers 7.1 - 7.3
            #a8 = probas[18:25] # Answers 8.1 - 8.7
            #a9 = probas[25:28] # Answers 9.1 - 9.3
            #a10 = probas[28:31] # Answers 10.1 - 10.3
            #a11 = probas[31:37] # Answers 11.1 - 11.6
            
            # galaxy classified as other if not in one of the other categories
            labels[line[0]] = 4 # other
            
            # if participants say there's something odd (question 6), classify as other
            if np.argmax(a6) == 0:
                continue
            
            else:
                # if most participants' answer to the first question is "smooth" ...
                if np.argmax(a1) == 0:
                    
                    # if completely round
                    if np.argmax(a7) == 0:
                        labels[line[0]] = 3 # round
                        
                    # if not ...
                    else:
                        labels[line[0]] = 2 # elliptical
                
                # if most participants' answer to the first question is "featrues or disk" ...
                elif np.argmax(a1) == 1:
                    # if most participants say there are spiral arms ...
                    
                    if np.argmax(a2) == 1 and np.argmax(a4) == 0:
                        labels[line[0]] = 1 # spiral
                        
                    # if not ...
                    else:
                        labels[line[0]] = 0 # disc
                
    return labels
        
def load_samples(handler, sample='training', grey_scale=False, as_vector=True):
    
    if sample == 'training':
        files_path = handler.training_images
    elif sample == 'validation':
        files_path = handler.validation_images
    elif sample == 'testing':
        files_path = handler.testing_images
    
    X = []
    y = []
    
    for file in files_path:
        
        try:
            img = os.path.join(handler.sample_paths[sample], 'Preprocessed', file)
            
            if grey_scale:
                img = io.imread(img, as_grey=True)
            else:
                img = io.imread(img)
            
            if as_vector:
                n_features = np.prod(img.shape)
                features = np.reshape(img, (1, n_features))
                X.append(features)
            else:
                X.append(img)
    
            id_ = handler.get_image_id(file)
            c = handler.find_label(id_)
            
            y.append(c)
            
        except:
            print('Unable to open '+ file)
            continue
            
    X = np.squeeze(np.array(X))
    y = np.array(y)
        
    return X, y
            

def compute_pca(X_train, n_comp, X_test=None):
    
    pca = PCA(n_components=n_comp, svd_solver='randomized', whiten=True)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    
    if X_test is None:        
        return X_train_pca, pca
    else:
        X_test_pca = pca.transform(X_test)
        return X_train_pca, X_test_pca, pca
    
def image_processing(images):
    n_images = len(images)
    mat = np.zeros(shape=(n_images, 3, 71, 71))
    for idx, img in enumerate(images):
        img = plt.imread(img).T
        mat[idx] = img
    return mat
    
def generate_batch(handler, sample='training'):
    if sample == 'training':
        files_path = handler.training_images
    elif sample == 'validation':
        files_path = handler.validation_images
    elif sample == 'testing':
        files_path = handler.testing_images
    
    while 1:
        for file in files_path:
            try:
                X = image_processing([handler.sample_paths[sample] + 'Preprocessed' + fname for fname in [file]])
                id_ = handler.get_image_id(file)
                
                if sample == 'testing':
                    yield (X)    
                else:
                    y = np.array(handler.find_label(id_))
                    y = np.reshape(y,(1,handler.output_size))
                    yield (X, y)
            except:
                next
                