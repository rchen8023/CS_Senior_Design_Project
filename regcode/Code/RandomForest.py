# Random Forest Classification Model

# For the student_cleaned_withACT.csv dataset



import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier



data = np.genfromtxt('student_cleaned_withACT.csv', delimiter=',', usecols=np.arange(0,163))

data = np.nan_to_num(data)



# Create the sample for the data

sample = data[1:,:]

[n,p] = sample.shape



# Stores the maximum number of values used in our training sample

max_num_train = 14123
# 70% for training



# Number of testing data

num_test = n - max_num_train



#Create a Random Forest Classifier

model = RandomForestClassifier(n_estimators=100, max_features='auto', n_jobs = -1)



num_train_trial = 10

num_trial = 10





# Setup arrays

error_train = np.zeros((num_trial, num_train_trial))

error_test = np.zeros((num_trial, num_train_trial))



probability_men = np.zeros((num_trial, num_train_trial))

probability_women = np.zeros((num_trial, num_train_trial))


probability_difference_test1 =np.zeros((num_trial, num_train_trial))

probability_difference_test2 =np.zeros((num_trial, num_train_trial))



false_positive_men = np.zeros((num_trial, num_train_trial))

false_positive_women = np.zeros((num_trial, num_train_trial))


false_positive_test1 = np.zeros((num_trial, num_train_trial))

false_positive_test2 = np.zeros((num_trial, num_train_trial))



demographic_parity_men = np.zeros((num_trial, num_train_trial))

demographic_parity_women = np.zeros((num_trial, num_train_trial))


demographic_parity_test1 = np.zeros((num_trial, num_train_trial))

demographic_parity_test2 = np.zeros((num_trial, num_train_trial))



training_instances = np.zeros(num_train_trial)



# Repeat over many random trials

for k in range(0, num_trial):

    print('new trial', k)
    

    # Shuffle the data each iteration

    np.random.shuffle(sample)

    sample_test = sample[max_num_train:,1:]

    label_test = sample[max_num_train:,0]

    

    # Count the number of men and women in the testing set

    num_men = sum(sample_test[:, 0])

    num_women = len(label_test) - num_men
    
    

    # Calculate the number of dropouts for each group (Differnce in Dropout Rate)

    num_men_dropout = len([e for e in range(0, num_test) if label_test[e] == 0 and sample_test[e, 0] == 1])

    num_women_dropout = len([e for e in range(0, num_test) if label_test[e] == 0 and sample_test[e, 0] == 1])

    

    # Increase the training size each time

    for i in range(0,num_train_trial):

        print('increase train size', i)
        

        # Increase the training size

        num_train = int(max_num_train/num_train_trial) * (i + 1)

        training_instances[i] = (num_train)



        # Split dataset into training set and test set

        sample_train = sample[0:num_train,1:]

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
        

        probability_difference_test1[k,i] = (abs(num_men_dropout/num_men - num_women_dropout/num_women))

        

        # Calculate the probability we pick someone to dropout and they do not dropout (False Positive Parity)

        num_men_incorrect = len([e for e in range(0, num_test) if label_test_predicted[e] == 0 and label_test[e] == 1 and sample_test[e, 0] == 1])

        num_women_incorrect = len([e for e in range(0, num_test) if label_test_predicted[e] == 0 and label_test[e] == 1 and sample_test[e, 0] == 0])

    

        false_positive_men[k,i] = (num_men_incorrect/num_men)

        false_positive_women[k,i] = (num_women_incorrect/num_women)
        

        false_positive_test1[k,i] = (abs(num_men_incorrect/num_men - num_women_incorrect/num_women))

        

        # Calculate the probability we predict someone will dropout between groups (Demographic Parity)

        num_men_predict = len([e for e in range(0, num_test) if label_test_predicted[e] == 0 and sample_test[e, 0] == 1])

        num_women_predict = len([e for e in range(0, num_test) if label_test_predicted[e] == 0 and sample_test[e, 0] == 0])

    

        demographic_parity_men[k,i] = (num_men_predict/num_men)

        demographic_parity_women[k,i] = (num_women_predict/num_women)
        

        demographic_parity_test1[k,i] = (abs(num_men_predict/num_men - num_women_predict/num_women))



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



# Plot the result

plt.figure()

plt.errorbar(x = training_instances, y = false_positive_test_ave1, yerr = [false_positive_test_err_lower1, false_positive_test_err_upper1], label='False Positive Parity', capsize=4, capthick = 2)

plt.errorbar(x = training_instances, y = demographic_parity_test_ave1, yerr = [demographic_parity_test_err_lower1, demographic_parity_test_err_upper1], label='Demographic Parity', capsize=4, capthick = 2)

plt.errorbar(x = training_instances, y = probability_difference_test_ave1, yerr = [probability_difference_test_err_lower1, probability_difference_test_err_upper1], label='Difference in True Dropout Rate', capsize=4, capthick = 2)

plt.errorbar(x = training_instances, y = error_test_ave, yerr = [error_test_err_lower, error_test_upper], label='Testing Error', capsize=4, capthick = 2)

plt.xlabel('Number of Training Instances')

plt.ylabel('Probability')

plt.title('Random Forest with 10 Trials (Gender)')

plt.legend()

plt.show()

