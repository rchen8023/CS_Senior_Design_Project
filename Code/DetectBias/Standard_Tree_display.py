


import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
#from sklearn.datasets import load_iris

#iris = load_iris()

data = np.genfromtxt('student-por_Cleaned.csv',delimiter=',',usecols=np.arange(0,46))
data = np.nan_to_num(data)

sample = data[1:,:]
[n,p] = sample.shape

num_train = int(1.0*n)

sample_train = sample[0:num_train,2:]
label_train = sample[0:num_train,0] 
        
sample_test = sample[num_train:,2:]
label_test = sample[num_train:,0]

clf = DecisionTreeClassifier(max_depth=3)
clf.fit(sample_train,label_train)

dot_data = StringIO()

export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
 
Image(graph.create_png())
