# Logistic Regression Classification Model
# For the harvardEdx MOOC dataset

# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegressionCV

data = np.genfromtxt('student-por_Cleaned.csv', delimiter=',', usecols=np.arange(0,46))
data = np.nan_to_num(data)

# Create the sample for the data
## The sample will hold the label in the first column, and the rest of the data in the other column
#sampleMat = np.asmatrix(data[1:,1])
#dataMat = np.asmatrix(data[1:,5:])
#sample = np.concatenate((sampleMat.T,dataMat),axis = 1)
#sample = np.asarray(sample)
#[n,p] = sample.shape

sample = data[1:,:]
#label = data[1:,0]
[n,p] = sample.shape

# Stores the maximum number of values used in our training sample
max_num_train = 500

# Number of testing data
num_test = n - max_num_train

#Create a Logistic Regression Classifier
model = LogisticRegressionCV(cv=5, solver='lbfgs', max_iter=500, tol=0.0001, n_jobs = 4)

num_train_trial = 10
num_trial = 10

# Setup arrays
error_train = np.zeros((num_trial, num_train_trial))
error_test = np.zeros((num_trial, num_train_trial))

probability_men = np.zeros((num_trial, num_train_trial))
probability_women = np.zeros((num_trial, num_train_trial))
probability_urban = np.zeros((num_trial, num_train_trial))
probability_rural = np.zeros((num_trial, num_train_trial))

probability_difference_test1 =np.zeros((num_trial, num_train_trial))
probability_difference_test2 =np.zeros((num_trial, num_train_trial))

false_positive_men = np.zeros((num_trial, num_train_trial))
false_positive_women = np.zeros((num_trial, num_train_trial))
false_positive_urban = np.zeros((num_trial, num_train_trial))
false_positive_rural = np.zeros((num_trial, num_train_trial))

false_positive_test1 = np.zeros((num_trial, num_train_trial))
false_positive_test2 = np.zeros((num_trial, num_train_trial))


demographic_parity_men = np.zeros((num_trial, num_train_trial))
demographic_parity_women = np.zeros((num_trial, num_train_trial))
demographic_parity_urban = np.zeros((num_trial, num_train_trial))
demographic_parity_rural = np.zeros((num_trial, num_train_trial))

demographic_parity_test1 = np.zeros((num_trial, num_train_trial))
demographic_parity_test2 = np.zeros((num_trial, num_train_trial))

# intersectional
probability_men_urban = np.zeros((num_trial, num_train_trial))
probability_women_rural = np.zeros((num_trial, num_train_trial))
probability_women_urban = np.zeros((num_trial, num_train_trial))
probability_men_rural = np.zeros((num_trial, num_train_trial))

probability_difference_test3 =np.zeros((num_trial, num_train_trial))
probability_difference_test4 =np.zeros((num_trial, num_train_trial))

false_positive_Mu = np.zeros((num_trial, num_train_trial))
false_positive_Wr = np.zeros((num_trial, num_train_trial))
false_positive_Wu = np.zeros((num_trial, num_train_trial))
false_positive_Mr = np.zeros((num_trial, num_train_trial))

false_positive_test3 = np.zeros((num_trial, num_train_trial))
false_positive_test4 = np.zeros((num_trial, num_train_trial))


demographic_parity_Mu = np.zeros((num_trial, num_train_trial))
demographic_parity_Wr = np.zeros((num_trial, num_train_trial))
demographic_parity_Wu = np.zeros((num_trial, num_train_trial))
demographic_parity_Mr = np.zeros((num_trial, num_train_trial))

demographic_parity_test3 = np.zeros((num_trial, num_train_trial))
demographic_parity_test4 = np.zeros((num_trial, num_train_trial))

training_instances = np.zeros(num_train_trial)

# Repeat over many random trials
for k in range(0, num_trial):
    print('new trial', k)
    
    # Shuffle the data each iteration
    np.random.shuffle(sample)
    sample_test = sample[max_num_train:,2:]
    label_test = sample[max_num_train:,0]
    
    # Count the number of men and women in the testing set
    num_men = sum(sample_test[:, 0])
    num_women = len(label_test) - num_men
    num_urban = sum(sample_test[:,0])
    num_rural = len(label_test) - num_urban
    
     # Calculate the number of dropouts for each group (Differnce in Dropout Rate)
    num_men_dropout = len([e for e in range(0, num_test) if label_test[e] == 0 and sample_test[e, 0] == 1])
    num_women_dropout = len([e for e in range(0, num_test) if label_test[e] == 0 and sample_test[e, 0] == 0])
    num_urban_dropout = len([e for e in range(0, num_test) if label_test[e] == 0 and sample_test[e, 16] == 1])
    num_rural_dropout = len([e for e in range(0, num_test) if label_test[e] == 0 and sample_test[e, 16] == 0])
    
    # intersectional
    # Count the number of men in urban, women in rural, women in urban, and men in rural in the testing set
    num_men_urban = len([e for e in range(0, num_test) if sample_test[e, 0] == 1 and sample_test[e, 16] == 1])
    num_women_rural = len([e for e in range(0, num_test) if sample_test[e, 0] == 0 and sample_test[e, 16] == 0])
    num_women_urban = len([e for e in range(0, num_test) if sample_test[e, 0] == 0 and sample_test[e, 16] == 1])
    num_men_rural = len([e for e in range(0, num_test) if sample_test[e, 0] == 1 and sample_test[e, 16] == 1])
    
    # Calculate the number of dropouts for each group (Differnce in Dropout Rate)
    num_Mu_dropout = len([e for e in range(0, num_test) if label_test[e] == 0 and sample_test[e, 0] == 1 and sample_test[e, 16] == 1])
    num_Wr_dropout = len([e for e in range(0, num_test) if label_test[e] == 0 and sample_test[e, 0] == 0 and sample_test[e, 16] == 0])
    num_Wu_dropout = len([e for e in range(0, num_test) if label_test[e] == 0 and sample_test[e, 0] == 0 and sample_test[e, 16] == 1])
    num_Mr_dropout = len([e for e in range(0, num_test) if label_test[e] == 0 and sample_test[e, 0] == 1 and sample_test[e, 16] == 1])

    # Increase the training size each time
    for i in range(0, num_train_trial):
        print('increase train size', i)
        
        # Increase the training size
        num_train = int(max_num_train/num_train_trial) * (i + 1)
        training_instances[i] = (num_train)
    
        # Split dataset into training set and test set
        sample_train = sample[0:num_train,2:]
        label_train = sample[0:num_train,0] 
        
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
        probability_urban[k,i] = (num_urban_dropout/num_urban)
        probability_rural[k,i] = (num_rural_dropout/num_rural)
        
        probability_difference_test1[k,i] = (abs(num_men_dropout/num_men - num_women_dropout/num_women))
        probability_difference_test2[k,i] = (abs(num_urban_dropout/num_urban - num_rural_dropout/num_rural))
        
        # Calculate the probability we pick someone to dropout and they do not dropout (False Positive Parity)
        num_men_incorrect = len([e for e in range(0, num_test) if label_test_predicted[e] == 0 and label_test[e] == 1 and sample_test[e, 0] == 1])
        num_women_incorrect = len([e for e in range(0, num_test) if label_test_predicted[e] == 0 and label_test[e] == 1 and sample_test[e, 0] == 0])
        num_urban_incorrect = len([e for e in range(0, num_test) if label_test_predicted[e] == 0 and label_test[e] == 1 and sample_test[e, 16] == 1])
        num_rural_incorrect = len([e for e in range(0, num_test) if label_test_predicted[e] == 0 and label_test[e] == 1 and sample_test[e, 16] == 0])
    
        false_positive_men[k,i] = (num_men_incorrect/num_men)
        false_positive_women[k,i] = (num_women_incorrect/num_women)
        false_positive_urban[k,i] = (num_urban_incorrect/num_urban)
        false_positive_rural[k,i] = (num_rural_incorrect/num_rural)
        
        false_positive_test1[k,i] = (abs(num_men_incorrect/num_men - num_women_incorrect/num_women))
        false_positive_test2[k,i] = (abs(num_urban_incorrect/num_urban - num_rural_incorrect/num_rural))
        
        # Calculate the probability we predict someone will dropout between groups (Demographic Parity)
        num_men_predict = len([e for e in range(0, num_test) if label_test_predicted[e] == 0 and sample_test[e, 0] == 1])
        num_women_predict = len([e for e in range(0, num_test) if label_test_predicted[e] == 0 and sample_test[e, 0] == 0])
        num_urban_predict = len([e for e in range(0, num_test) if label_test_predicted[e] == 0 and sample_test[e, 16] == 1])
        num_rural_predict = len([e for e in range(0, num_test) if label_test_predicted[e] == 0 and sample_test[e, 16] == 0])
    
        demographic_parity_men[k,i] = (num_men_predict/num_men)
        demographic_parity_women[k,i] = (num_women_predict/num_women)
        demographic_parity_urban[k,i] = (num_urban_predict/num_urban)
        demographic_parity_rural[k,i] = (num_rural_predict/num_rural)
        
        demographic_parity_test1[k,i] = (abs(num_men_predict/num_men - num_women_predict/num_women))
        demographic_parity_test2[k,i] = (abs(num_urban_predict/num_urban - num_rural_predict/num_rural))
        
        # intersectional
        # Count dropout
        probability_men_urban[k,i] = (num_Mu_dropout/num_men_urban)
        probability_women_rural[k,i] = (num_Wr_dropout/num_women_rural)
        probability_women_urban[k,i] = (num_Wu_dropout/num_women_urban)
        probability_men_rural[k,i] = (num_Mr_dropout/num_men_rural)
        
        probability_difference_test3[k,i] = (abs(num_Mu_dropout/num_men_urban - num_Wr_dropout/num_women_rural))
        probability_difference_test4[k,i] = (abs(num_Wu_dropout/num_women_urban - num_Mr_dropout/num_men_rural))
        
        # Calculate the probability we pick someone to dropout and they do not dropout (False Positive Parity)
        num_Mu_incorrect = len([e for e in range(0, num_test) if label_test_predicted[e] == 0 and label_test[e] == 1 and sample_test[e, 0] == 1 and sample_test[e, 16] == 1])
        num_Wr_incorrect = len([e for e in range(0, num_test) if label_test_predicted[e] == 0 and label_test[e] == 1 and sample_test[e, 0] == 0 and sample_test[e, 16] == 0])
        num_Wu_incorrect = len([e for e in range(0, num_test) if label_test_predicted[e] == 0 and label_test[e] == 1 and sample_test[e, 0] == 0 and sample_test[e, 16] == 1])
        num_Mr_incorrect = len([e for e in range(0, num_test) if label_test_predicted[e] == 0 and label_test[e] == 1 and sample_test[e, 0] == 1 and sample_test[e, 16] == 1])
    
        false_positive_Mu[k,i] = (num_Mu_incorrect/num_men_urban)
        false_positive_Wr[k,i] = (num_Wr_incorrect/num_women_rural)
        false_positive_Wu[k,i] = (num_Wu_incorrect/num_women_urban)
        false_positive_Mr[k,i] = (num_Mr_incorrect/num_men_rural)
        
        false_positive_test3[k,i] = (abs(num_Mu_incorrect/num_men_urban - num_Wr_incorrect/num_women_rural))
        false_positive_test4[k,i] = (abs(num_Wu_incorrect/num_women_urban - num_Mr_incorrect/num_men_rural))
        
        # Calculate the probability we predict someone will dropout between groups (Demographic Parity)
        num_Mu_predict = len([e for e in range(0, num_test) if label_test_predicted[e] == 0 and sample_test[e, 0] == 1 and sample_test[e, 16] == 1])
        num_Wr_predict = len([e for e in range(0, num_test) if label_test_predicted[e] == 0 and sample_test[e, 0] == 0 and sample_test[e, 16] == 0])
        num_Wu_predict = len([e for e in range(0, num_test) if label_test_predicted[e] == 0 and sample_test[e, 0] == 0 and sample_test[e, 16] == 1])
        num_Mr_predict = len([e for e in range(0, num_test) if label_test_predicted[e] == 0 and sample_test[e, 0] == 1 and sample_test[e, 16] == 1])
    
        demographic_parity_Mu[k,i] = (num_Mu_predict/num_men_urban)
        demographic_parity_Wr[k,i] = (num_Wr_predict/num_women_rural)
        demographic_parity_Wu[k,i] = (num_Wu_predict/num_women_urban)
        demographic_parity_Mr[k,i] = (num_Mr_predict/num_men_rural)
        
        demographic_parity_test3[k,i] = (abs(num_Mu_predict/num_men_urban - num_Wr_predict/num_women_rural))
        demographic_parity_test4[k,i] = (abs(num_Wu_predict/num_women_urban - num_Mr_predict/num_men_rural))

# Gender
# Average all of the runs of our test
false_positive_test_ave1 = false_positive_test1.mean(axis = 0)
demographic_parity_test_ave1 = demographic_parity_test1.mean(axis = 0)
probability_difference_test_ave1 = probability_difference_test1.mean(axis = 0)
error_test_ave = error_test.mean(axis = 0)

# Calculate the error for each measurement
false_positive_test_err_upper1 = false_positive_test1.max(axis = 0) - false_positive_test1.mean(axis = 0)
false_positive_test_err_lower1 = abs(false_positive_test1.min(axis = 0) - false_positive_test1.mean(axis = 0))

demographic_parity_test_err_upper1 = demographic_parity_test1.max(axis = 0) - demographic_parity_test1.mean(axis = 0)
demographic_parity_test_err_lower1 = abs(demographic_parity_test1.min(axis = 0) - demographic_parity_test1.mean(axis = 0))

error_test_upper = error_test.max(axis = 0) - error_test.mean(axis = 0)
error_test_err_lower = abs(error_test.min(axis = 0) - error_test.mean(axis = 0))

probability_difference_test_err_upper1 = probability_difference_test1.max(axis = 0) - probability_difference_test1.mean(axis = 0)
probability_difference_test_err_lower1 = abs(probability_difference_test1.min(axis = 0) - probability_difference_test1.mean(axis = 0))

# Address
# Average all of the runs of our test
false_positive_test_ave2 = false_positive_test2.mean(axis = 0)
demographic_parity_test_ave2 = demographic_parity_test2.mean(axis = 0)
probability_difference_test_ave2 = probability_difference_test2.mean(axis = 0)

# Calculate the error for each measurement
false_positive_test_err_upper2 = false_positive_test2.max(axis = 0) - false_positive_test2.mean(axis = 0)
false_positive_test_err_lower2 = abs(false_positive_test2.min(axis = 0) - false_positive_test2.mean(axis = 0))

demographic_parity_test_err_upper2 = demographic_parity_test2.max(axis = 0) - demographic_parity_test2.mean(axis = 0)
demographic_parity_test_err_lower2 = abs(demographic_parity_test2.min(axis = 0) - demographic_parity_test2.mean(axis = 0))

probability_difference_test_err_upper2 = probability_difference_test2.max(axis = 0) - probability_difference_test2.mean(axis = 0)
probability_difference_test_err_lower2 = abs(probability_difference_test2.min(axis = 0) - probability_difference_test2.mean(axis = 0))

# Men_urban vs. Women_rural
# Average all of the runs of our test
false_positive_test_ave3 = false_positive_test3.mean(axis = 0)
demographic_parity_test_ave3 = demographic_parity_test3.mean(axis = 0)
probability_difference_test_ave3 = probability_difference_test3.mean(axis = 0)

# Calculate the error for each measurement
false_positive_test_err_upper3 = false_positive_test3.max(axis = 0) - false_positive_test3.mean(axis = 0)
false_positive_test_err_lower3 = abs(false_positive_test3.min(axis = 0) - false_positive_test3.mean(axis = 0))

demographic_parity_test_err_upper3 = demographic_parity_test3.max(axis = 0) - demographic_parity_test3.mean(axis = 0)
demographic_parity_test_err_lower3 = abs(demographic_parity_test3.min(axis = 0) - demographic_parity_test3.mean(axis = 0))

probability_difference_test_err_upper3 = probability_difference_test3.max(axis = 0) - probability_difference_test3.mean(axis = 0)
probability_difference_test_err_lower3 = abs(probability_difference_test3.min(axis = 0) - probability_difference_test3.mean(axis = 0))

# Women_urban vs. Men_rural
# Average all of the runs of our test
false_positive_test_ave4 = false_positive_test4.mean(axis = 0)
demographic_parity_test_ave4 = demographic_parity_test4.mean(axis = 0)
probability_difference_test_ave4 = probability_difference_test4.mean(axis = 0)

# Calculate the error for each measurement
false_positive_test_err_upper4 = false_positive_test4.max(axis = 0) - false_positive_test4.mean(axis = 0)
false_positive_test_err_lower4 = abs(false_positive_test4.min(axis = 0) - false_positive_test4.mean(axis = 0))

demographic_parity_test_err_upper4 = demographic_parity_test4.max(axis = 0) - demographic_parity_test4.mean(axis = 0)
demographic_parity_test_err_lower4 = abs(demographic_parity_test4.min(axis = 0) - demographic_parity_test4.mean(axis = 0))

probability_difference_test_err_upper4 = probability_difference_test4.max(axis = 0) - probability_difference_test4.mean(axis = 0)
probability_difference_test_err_lower4 = abs(probability_difference_test4.min(axis = 0) - probability_difference_test4.mean(axis = 0))


# Plot the result
plt.figure()
plt.errorbar(x = training_instances, y = false_positive_test_ave1, yerr = [false_positive_test_err_lower1, false_positive_test_err_upper1], label='False Positive Parity', capsize=4, capthick = 2)
plt.errorbar(x = training_instances, y = demographic_parity_test_ave1, yerr = [demographic_parity_test_err_lower1, demographic_parity_test_err_upper1], label='Demographic Parity', capsize=4, capthick = 2)
plt.errorbar(x = training_instances, y = probability_difference_test_ave1, yerr = [probability_difference_test_err_lower1, probability_difference_test_err_upper1], label='Difference in True Dropout Rate', capsize=4, capthick = 2)
plt.errorbar(x = training_instances, y = error_test_ave, yerr = [error_test_err_lower, error_test_upper], label='Testing Error', capsize=4, capthick = 2)
plt.xlabel('Number of Training Instances')
plt.ylabel('Probability')
plt.title('LR with 10 Trials (Gender)')
plt.legend()
plt.show()

# Plot the result
plt.figure()
plt.errorbar(x = training_instances, y = false_positive_test_ave2, yerr = [false_positive_test_err_lower2, false_positive_test_err_upper2], label='False Positive Parity', capsize=4, capthick = 2)
plt.errorbar(x = training_instances, y = demographic_parity_test_ave2, yerr = [demographic_parity_test_err_lower2, demographic_parity_test_err_upper2], label='Demographic Parity', capsize=4, capthick = 2)
plt.errorbar(x = training_instances, y = probability_difference_test_ave2, yerr = [probability_difference_test_err_lower2, probability_difference_test_err_upper2], label='Difference in True Dropout Rate', capsize=4, capthick = 2)
plt.errorbar(x = training_instances, y = error_test_ave, yerr = [error_test_err_lower, error_test_upper], label='Testing Error', capsize=4, capthick = 2)
plt.xlabel('Number of Training Instances')
plt.ylabel('Probability')
plt.title('LR with 10 Trials (Address)')
plt.legend()
plt.show()

# Plot the result
plt.figure()
plt.errorbar(x = training_instances, y = false_positive_test_ave3, yerr = [false_positive_test_err_lower3, false_positive_test_err_upper3], label='False Positive Parity', capsize=4, capthick = 2)
plt.errorbar(x = training_instances, y = demographic_parity_test_ave3, yerr = [demographic_parity_test_err_lower3, demographic_parity_test_err_upper3], label='Demographic Parity', capsize=4, capthick = 2)
plt.errorbar(x = training_instances, y = probability_difference_test_ave3, yerr = [probability_difference_test_err_lower3, probability_difference_test_err_upper3], label='Difference in True Dropout Rate', capsize=4, capthick = 2)
plt.errorbar(x = training_instances, y = error_test_ave, yerr = [error_test_err_lower, error_test_upper], label='Testing Error', capsize=4, capthick = 2)
plt.xlabel('Number of Training Instances')
plt.ylabel('Probability')
plt.title('LR with 10 Trials (Mu vs. Wr)')
plt.legend()
plt.show()

# Plot the result
plt.figure()
plt.errorbar(x = training_instances, y = false_positive_test_ave4, yerr = [false_positive_test_err_lower4, false_positive_test_err_upper4], label='False Positive Parity', capsize=4, capthick = 2)
plt.errorbar(x = training_instances, y = demographic_parity_test_ave4, yerr = [demographic_parity_test_err_lower4, demographic_parity_test_err_upper4], label='Demographic Parity', capsize=4, capthick = 2)
plt.errorbar(x = training_instances, y = probability_difference_test_ave4, yerr = [probability_difference_test_err_lower4, probability_difference_test_err_upper4], label='Difference in True Dropout Rate', capsize=4, capthick = 2)
plt.errorbar(x = training_instances, y = error_test_ave, yerr = [error_test_err_lower, error_test_upper], label='Testing Error', capsize=4, capthick = 2)
plt.xlabel('Number of Training Instances')
plt.ylabel('Probability')
plt.title('LR with 10 Trials (Wu vs. Mr)')
plt.legend()
plt.show()
