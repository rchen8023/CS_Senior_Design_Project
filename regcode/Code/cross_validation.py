from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

num_train = 15000
max_num_train= 15000

data = np.genfromtxt('student_cleaned_withACT.csv', delimiter=',')
data = np.nan_to_num(data)
sample = data[1:,1:]
[n,p] = sample.shape

sample_train = sample[0:num_train,:-1]
label_train = sample[0:num_train,-1] 

sample_test = sample[max_num_train:,:-1]
label_test = sample[max_num_train:,-1]

clf = RandomForestClassifier(bootstrap = True, max_features='auto', n_jobs = -1)

param_grid = {
        'n_estimators': [100],
        'max_depth': [10,20,30,40],
        'class_weight': [{0:1,1:10},{0:1,1:5},{0:1,1:2}]
}

grid_clf = GridSearchCV(clf, param_grid, cv=5, n_jobs = -1)
grid_clf.fit(sample_train, label_train)

print(grid_clf.best_params_)