# AdaBoost Classification Model
# For the harvardEdx MOOC dataset

# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

data = np.genfromtxt('harvardEdxCategoricalCleanTest.csv', delimiter=',', usecols=np.arange(0,68))
data = np.nan_to_num(data)

# Create the sample for the data
# The sample will hold the label in the first column, and the rest of the data in the other column
sampleMat = np.asmatrix(data[1:,1])
dataMat = np.asmatrix(data[1:,5:])
data = np.concatenate((sampleMat.T,dataMat),axis = 1)
data = np.asarray(data)
[n,p] = data.shape

# Stores the maximum number of values used in our training sample
max_num_train = 5000

# Number of testing data
num_test = n - max_num_train

#Create an AdaBoost Classifier
model = AdaBoostClassifier(n_estimators=100)

num_train_trial = 10
num_trial = 5

# Setup arrays
error_train = np.zeros((num_trial, num_train_trial))
error_test = np.zeros((num_trial, num_train_trial))

probability_men = np.zeros((num_trial, num_train_trial))
probability_women = np.zeros((num_trial, num_train_trial))
probability_difference_test =np.zeros((num_trial, num_train_trial))

false_positive_men = np.zeros((num_trial, num_train_trial))
false_positive_women = np.zeros((num_trial, num_train_trial))
false_positive_test = np.zeros((num_trial, num_train_trial))

demographic_parity_men = np.zeros((num_trial, num_train_trial))
demographic_parity_women = np.zeros((num_trial, num_train_trial))
demographic_parity_test = np.zeros((num_trial, num_train_trial))
training_instances = np.zeros(num_train_trial)

# Repeat over many random trials
for k in range(0, num_trial):
    print('new trial', k)
    
    # Shuffle the data each iteration
    np.random.shuffle(data)
    sample_test = data[max_num_train:,1:]
    label_test = data[max_num_train:,0]
    
    # Count the number of men and women in the testing set
    num_men = sum(sample_test[:, -1])
    num_women = sum(sample_test[:, -2])
    
    # Calculate the number of dropouts for each group (Differnce in Dropout Rate)
    num_men_dropout = len([e for e in range(0, num_test) if label_test[e] == 0 and sample_test[e, -1] == 1])
    num_women_dropout = len([e for e in range(0, num_test) if label_test[e] == 0 and sample_test[e, -2] == 1])
    
    # Increase the training size each time
    for i in range(0, num_train_trial):
        print('increase train size', i)
        
        # Increase the training size
        num_train = int(max_num_train/num_train_trial) * (i + 1)
        training_instances[i] = (num_train)
    
        # Split dataset into training set and test set
        sample_train = data[0:num_train,1:]
        label_train = data[0:num_train,0] 
        
    	# Fit the model to our training set
        model.fit(sample_train,label_train)
        
        # Get predicted values from values
        label_train_predicted = model.predict(sample_train)
        label_test_predicted = model.predict(sample_test)
        
        # Calculate the error in each prediction
        error_train[k,i] = 1 - (accuracy_score(label_train, label_train_predicted))
        error_test[k,i] = 1 - (accuracy_score(label_test, label_test_predicted))
        
        # Count dropout
        probability_men[k,i] = (num_men_dropout/num_men)
        probability_women[k,i] = (num_women_dropout/num_women)
        probability_difference_test[k,i] = (abs(num_men_dropout/num_men - num_women_dropout/num_women))
        
        # Calculate the probability we pick someone to dropout and they do not dropout (False Positive Parity)
        num_men_incorrect = len([e for e in range(0, num_test) if label_test_predicted[e] == 0 and label_test[e] == 1 and sample_test[e, -1] == 1])
        num_women_incorrect = len([e for e in range(0, num_test) if label_test_predicted[e] == 0 and label_test[e] == 1 and sample_test[e, -2] == 1])
        
        false_positive_men[k,i] = (num_men_incorrect/num_men)
        false_positive_women[k,i] = (num_women_incorrect/num_women)
        false_positive_test[k,i] = (abs(num_men_incorrect/num_men - num_women_incorrect/num_women))
        
        # Calculate the probability we predict someone will dropout between groups (Demographic Parity)
        num_men_predict = len([e for e in range(0, num_test) if label_test_predicted[e] == 0 and sample_test[e, -1] == 1])
        num_women_predict = len([e for e in range(0, num_test) if label_test_predicted[e] == 0 and sample_test[e, -2] == 1])
        
        demographic_parity_men[k,i] = (num_men_predict/num_men)
        demographic_parity_women[k,i] = (num_women_predict/num_women)
        demographic_parity_test[k,i] = (abs(num_men_predict/num_men - num_women_predict/num_women))

# Average all of the runs of our test
false_positive_test_ave = false_positive_test.mean(axis = 0)
demographic_parity_test_ave = demographic_parity_test.mean(axis = 0)
probability_difference_test_ave = probability_difference_test.mean(axis = 0)
error_test_ave = error_test.mean(axis = 0)

# Calculate the error for each measurement
false_positive_test_err_upper = false_positive_test.max(axis = 0) - false_positive_test.mean(axis = 0)
false_positive_test_err_lower = abs(false_positive_test.min(axis = 0) - false_positive_test.mean(axis = 0))

demographic_parity_test_err_upper = demographic_parity_test.max(axis = 0) - demographic_parity_test.mean(axis = 0)
demographic_parity_test_err_lower = abs(demographic_parity_test.min(axis = 0) - demographic_parity_test.mean(axis = 0))

error_test_upper = error_test.max(axis = 0) - error_test.mean(axis = 0)
error_test_err_lower = abs(error_test.min(axis = 0) - error_test.mean(axis = 0))

probability_difference_test_err_upper = probability_difference_test.max(axis = 0) - probability_difference_test.mean(axis = 0)
probability_difference_test_err_lower = abs(probability_difference_test.min(axis = 0) - probability_difference_test.mean(axis = 0))

# Plot the result
plt.figure()
plt.errorbar(x = training_instances, y = false_positive_test_ave, yerr = [false_positive_test_err_lower, false_positive_test_err_upper], label='False Positive Parity', capsize=4, capthick = 2)
plt.errorbar(x = training_instances, y = demographic_parity_test_ave, yerr = [demographic_parity_test_err_lower, demographic_parity_test_err_upper], label='Demographic Parity', capsize=4, capthick = 2)
plt.errorbar(x = training_instances, y = probability_difference_test_ave, yerr = [probability_difference_test_err_lower, probability_difference_test_err_upper], label='Difference in True Dropout Rate', capsize=4, capthick = 2)
plt.errorbar(x = training_instances, y = error_test_ave, yerr = [error_test_err_lower, error_test_upper], label='Testing Error', capsize=4, capthick = 2)
plt.xlabel('Number of Training Instances')
plt.ylabel('Probability')
plt.title('AdaBoost with 10 Trials')
plt.legend()
plt.show()