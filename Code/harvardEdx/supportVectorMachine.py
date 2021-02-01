import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC

data = np.genfromtxt('harvardEdxCategoricalClean.csv', delimiter=',', usecols=np.arange(0,68))

data = np.nan_to_num(data)

# Create the sample for the data
sample = data[1:,5:]
label = data[1:,1]
[n,p] = sample.shape

# Setup arrays
error_train = []
error_test = []

probability_men = []
probability_women = []
probability_difference_test = []

false_positive_men = []
false_positive_women = []
false_positive_test = []

demographic_parity_men = []
demographic_parity_women = []
demographic_parity_test = []

training_instances = []

#Create a Logistic Regression Classifier
model = SVC(gamma='auto', kernel = 'sigmoid')

for i in range(0,5):
    num_train = 5000 * (i + 1)
    training_instances.append(num_train)

    # Split dataset into training set and test set
    sample_train = sample[0:num_train,:]
    label_train = label[0:num_train] 
    sample_test = sample[num_train:,:]
    label_test = label[num_train:]
    
	# Fit the model to our training set
    model.fit(sample_train,label_train)
    
    # Get predicted values from values
    label_train_predicted = model.predict(sample_train)
    label_test_predicted = model.predict(sample_test)
    
    # Calculate the error in each prediction
    error_train.append(mean_squared_error(label_train, label_train_predicted))
    error_test.append(mean_squared_error(label_test, label_test_predicted))

    # Count the number of men and women in the testing set
    num_men = sum(sample_test[:, -1])
    num_women = sum(sample_test[:, -2])
    
    # Calculate the number of dropouts for each group (Differnce in Dropout Rate)
    num_men_dropout = len([e for e in range(0, n - num_train) if label_test[e] == 0 and sample_test[e, -1] == 1])
    num_women_dropout = len([e for e in range(0, n - num_train) if label_test[e] == 0 and sample_test[e, -2] == 1])
    
    probability_men.append(num_men_dropout/num_men)
    probability_women.append(num_women_dropout/num_women)
    
    probability_difference_test.append(abs(num_men_dropout/num_men - num_women_dropout/num_women))
    
    # Calculate the probability we pick someone to dropout and they do not dropout (False Positive Parity)
    num_men_incorrect = len([e for e in range(0, n - num_train) if label_test_predicted[e] == 0 and label_test[e] == 1 and sample_test[e, -1] == 1])
    num_women_incorrect = len([e for e in range(0, n - num_train) if label_test_predicted[e] == 0 and label_test[e] == 1 and sample_test[e, -2] == 1])
    
    false_positive_men.append(num_men_incorrect/num_men)
    false_positive_women.append(num_women_incorrect/num_women)
    
    false_positive_test.append(abs(num_men_incorrect/num_men - num_women_incorrect/num_women))
    
    # Calculate the probability we predict someone will dropout between groups (Demographic Parity)
    num_men_predict = len([e for e in range(0, n - num_train) if label_test_predicted[e] == 0 and sample_test[e, -1] == 1])
    num_women_predict = len([e for e in range(0, n - num_train) if label_test_predicted[e] == 0 and sample_test[e, -2] == 1])
    
    demographic_parity_men.append(num_men_predict/num_men)
    demographic_parity_women.append(num_women_predict/num_women)
    
    demographic_parity_test.append(abs(num_men_predict/num_men - num_women_predict/num_women))

# Plot the result
plt.plot(training_instances, false_positive_test, label='False Positives Parity')
plt.plot(training_instances, demographic_parity_test, label='Demographic Parity')
plt.plot(training_instances, probability_difference_test, label='Difference in True Dropout Rate')
plt.plot(training_instances, error_test, label='Testing Error')
plt.xlabel('Number of Training Instances')
plt.ylabel('Probability')
plt.legend()
plt.show()