import matplotlib.pylab as plt
import data_utils
import numpy as np
from time import time
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

projectFolder = 'D:/Data Science/NYU Data Science/DS-GA 1003 Project'
data_dir_path = projectFolder + '/Data'

### --- Create Training / Validation and Testing Samples
sample_fractions = [0.6, 0.3, 0.1] # training / validation / testing
input_size = (71,71)
output_size = 1

n_classes = 5
classNames = {0: 'Disc',
              1: 'Spiral',
              2: 'Elliptical',
              3: 'Round',
              4: 'Other'}

n_comp = 400

handler = data_utils.data_handler(data_dir_path, sample_fractions=sample_fractions, 
                              input_size=input_size, labels_type='classes', 
                              output_size=output_size, normalize_input=False, 
                              create_samples_bool=False, preprocess_bool=False, 
                              crp_factor=2, ds_factor=3)
    
### Load data
X_train, y_train = data_utils.load_samples(handler, 'training', grey_scale=True)    
X_val, y_val = data_utils.load_samples(handler, 'validation', grey_scale=True)    

### Perform PCA
X_train_pca, X_val_pca, pca = data_utils.compute_pca(
        X_train=X_train, n_comp=n_comp, X_test=X_val)

### Train an SVM classification model
param_grid = {'C': 10.**np.arange(-4, 4, 1),
              'gamma': 10.**np.arange(-4, 4, 1), }
clf = GridSearchCV(svm.SVC(kernel='rbf'), param_grid)

t0 = time()
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print(clf.best_estimator_)

# #############################################################################
# Quantitative evaluation of the model quality on the test set
y_pred = clf.predict(X_val_pca)
print(metrics.classification_report(y_val, y_pred, target_names=classNames.values))
print(metrics.confusion_matrix(y_val, y_pred, labels=range(n_classes)))


# #############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()