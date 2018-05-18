import matplotlib.pylab as plt
import seaborn as sns
import data_utils
import numpy as np
from time import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

projectFolder = 'D:\\Data Science\\NYU Data Science\\DS-GA 1003 Project Smaller Sample'
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

n_comp = 200

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

### Train an one vs. all Logistic Regression model
param_grid = {"n_neighbors": np.arange(1, 31, 2)}

knn = KNeighborsClassifier(metric="euclidean") 
clf = GridSearchCV(estimator=knn, param_grid=param_grid, cv= 10)

t0 = time()
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print(clf.best_estimator_)

acc_score = [x[1] for x in clf.grid_scores_]

plt.figure(figsize=(6, 4))
ax = plt.gca()
ax.plot(clf.param_grid['n_neighbors'], acc_score)
plt.xlabel('Number of Neighbors (N)')
plt.ylabel('Validation Accuracy Score')
plt.title('Validation Accuracy Score as a function of N')
plt.axis('tight')

knn = KNeighborsClassifier(metric="euclidean",n_neighbors=10) 
knn.fit(X_train_pca, y_train)

y_pred = knn.predict(X_val_pca)
print(metrics.classification_report(y_val, y_pred, target_names=classNames.values()))

conf_knn = metrics.confusion_matrix(y_val, y_pred)

sns.heatmap(conf_knn, cmap='hot')
plt.title('KNN Confusion Matrix', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.yticks(range(n_classes), classNames.values(), fontsize=10, rotation='horizontal')
plt.xticks(range(n_classes), classNames.values(), fontsize=10, rotation='vertical')

accuracy = np.sum(y_val == y_pred) / len(y_val)



