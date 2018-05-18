import matplotlib.pylab as plt
import seaborn as sns
import data_utils
import numpy as np
from time import time
from sklearn.linear_model import LogisticRegression
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
param_grid = {'C': 10.**np.arange(-8, 8, 0.5)}

clf = GridSearchCV(LogisticRegression(multi_class='ovr'), param_grid, scoring="accuracy")

t0 = time()
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print(clf.best_estimator_)

acc_score = [x[1] for x in clf.grid_scores_]

plt.figure(figsize=(6, 4))
ax = plt.gca()
ax.plot(np.log(clf.param_grid['C'])/np.log(10), acc_score)
plt.xlabel('C (Log)')
plt.ylabel('Validation Accuracy Score')
plt.title('Validation Accuracy Score as a function of the Parameter C')
plt.axis('tight')

lr = LogisticRegression(multi_class='ovr', C=0.01)
lr.fit(X_train_pca, y_train)

y_pred = lr.predict(X_val_pca)
print(metrics.classification_report(y_val, y_pred, target_names=classNames.values()))

conf_lr = metrics.confusion_matrix(y_val, y_pred)

sns.heatmap(conf_lr, cmap='hot')
plt.title('One vs. All Logistic Regression', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.yticks(range(n_classes), classNames.values(), fontsize=10, rotation='horizontal')
plt.xticks(range(n_classes), classNames.values(), fontsize=10, rotation='vertical')

accuracy = np.sum(y_val == y_pred) / len(y_val)



