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

handler = data_utils.data_handler(data_dir_path, sample_fractions=sample_fractions, 
                              input_size=input_size, labels_type='classes', 
                              output_size=output_size, normalize_input=False, 
                              create_samples_bool=False, preprocess_bool=False, 
                              crp_factor=2, ds_factor=3)
    
### Load data
X_train, y_train = data_utils.load_features(handler, 'training')    
X_val, y_val = data_utils.load_features(handler, 'validation')   

### Train an one vs. all Logistic Regression model
param_grid = {"n_neighbors": np.arange(1, 31, 2)}

knn = KNeighborsClassifier(metric="euclidean") 
clf = GridSearchCV(estimator=knn, param_grid=param_grid, cv= 10)

t0 = time()
clf = clf.fit(X_train, y_train)
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

knn = KNeighborsClassifier(metric="euclidean",n_neighbors=15) 
knn.fit(X_train, y_train)

y_pred = knn.predict(X_val)
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



