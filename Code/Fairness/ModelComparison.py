# Random Forest Model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from extraRandomForestWithRidgeRegression import FairExtraRandomForestClassifierRidge
from extraFairRandomForest import FairExtraRandomForestClassifier


data = np.genfromtxt('student-por_Cleaned.csv', delimiter=',', usecols=np.arange(0,46))
data = np.nan_to_num(data)
sample = data[1:,:]

[n,p] = sample.shape

# Stores the maximum number of values used in our training sample
max_num_train = 500

# Number of testing data
num_test = n - max_num_train

ridge_model = FairExtraRandomForestClassifierRidge(gender_fair = True, address_fair = False, max_depth = 23, num_trees = 500)
standard_model = FairExtraRandomForestClassifier(gender_fair = True, address_fair = False, max_depth = 23, num_trees = 500)

num_train_trial = 10
num_trial = 10

# Setup arrays
error_train_ridge = np.zeros((num_trial, num_train_trial))
error_test_ridge  = np.zeros((num_trial, num_train_trial))

error_train_standard = np.zeros((num_trial, num_train_trial))
error_test_standard  = np.zeros((num_trial, num_train_trial))

probability_difference_test1 =np.zeros((num_trial, num_train_trial))
probability_difference_test2 =np.zeros((num_trial, num_train_trial))

demographic_parity_test1_ridge = np.zeros((num_trial, num_train_trial))
demographic_parity_test2_ridge = np.zeros((num_trial, num_train_trial))
demographic_parity_test1_standard = np.zeros((num_trial, num_train_trial))
demographic_parity_test2_standard = np.zeros((num_trial, num_train_trial))

# intersectional
probability_difference_test3 =np.zeros((num_trial, num_train_trial))
probability_difference_test4 =np.zeros((num_trial, num_train_trial))

probability_difference_testa1 = np.zeros((num_trial,num_train_trial))
probability_difference_testa2 = np.zeros((num_trial,num_train_trial))
probability_difference_testa3 = np.zeros((num_trial,num_train_trial))
probability_difference_testa4 = np.zeros((num_trial,num_train_trial))

demographic_parity_test3 = np.zeros((num_trial, num_train_trial))
demographic_parity_test4 = np.zeros((num_trial, num_train_trial))

demographic_parity_testa1 = np.zeros((num_trial, num_train_trial))
demographic_parity_testa2 = np.zeros((num_trial, num_train_trial))
demographic_parity_testa3 = np.zeros((num_trial, num_train_trial))
demographic_parity_testa4 = np.zeros((num_trial, num_train_trial))

training_instances = np.zeros(num_train_trial)

# Repeat over many random trials
for k in range(0, num_trial):
    print('new trial', k)
    
    # Shuffle the data each iteration
    np.random.shuffle(sample)
    sample_test = sample[max_num_train:,2:]
    label_test = sample[max_num_train:,0]
    
    # Count the number of men and women in the testing set
    num_women = sum(sample_test[:, 0])
    num_men = len(label_test) - num_women
    num_urban = sum(sample_test[:, 16])
    num_rural = len(label_test) - num_urban
    
     # Calculate the number of dropouts for each group (Differnce in Dropout Rate)
    num_men_dropout = len([e for e in range(0, num_test) if label_test[e] == 0 and sample_test[e, 0] == 0])
    num_women_dropout = len([e for e in range(0, num_test) if label_test[e] == 0 and sample_test[e, 0] == 1])
    num_urban_dropout = len([e for e in range(0, num_test) if label_test[e] == 0 and sample_test[e, 16] == 1])
    num_rural_dropout = len([e for e in range(0, num_test) if label_test[e] == 0 and sample_test[e, 16] == 0])
    
    # intersectional
    # Count the number of men in urban, women in rural, women in urban, and men in rural in the testing set
    num_men_urban = len([e for e in range(0, num_test) if sample_test[e, 0] == 0 and sample_test[e, 16] == 1])
    num_women_rural = len([e for e in range(0, num_test) if sample_test[e, 0] == 1 and sample_test[e, 16] == 0])
    num_women_urban = len([e for e in range(0, num_test) if sample_test[e, 0] == 1 and sample_test[e, 16] == 1])
    num_men_rural = len([e for e in range(0, num_test) if sample_test[e, 0] == 0 and sample_test[e, 16] == 0])
    
    # Calculate the number of dropouts for each group (Differnce in Dropout Rate)
    num_Mu_dropout = len([e for e in range(0, num_test) if label_test[e] == 0 and sample_test[e, 0] == 0 and sample_test[e, 16] == 1])
    num_Wr_dropout = len([e for e in range(0, num_test) if label_test[e] == 0 and sample_test[e, 0] == 1 and sample_test[e, 16] == 0])
    num_Wu_dropout = len([e for e in range(0, num_test) if label_test[e] == 0 and sample_test[e, 0] == 1 and sample_test[e, 16] == 1])
    num_Mr_dropout = len([e for e in range(0, num_test) if label_test[e] == 0 and sample_test[e, 0] == 0 and sample_test[e, 16] == 0])

    # Increase the number of base models
    for i in range(0, num_train_trial):
        print('increase ensemble size', i)
        
        # Increase the training size
        num_train = 500
        num_models = 10 * (i + 1)
        training_instances[i] = num_models
    
        # Split dataset into training set and test set
        sample_train = sample[0:num_train,2:]
        label_train = sample[0:num_train,0]  
        
    	# Fit the model to our training set
        ridge_model.fit(sample_train,label_train, num_fair_trees = num_models)
        standard_model.fit(sample_train,label_train, num_fair_trees = num_models)
        
        # Get predicted values from values
        label_train_predicted_ridge = ridge_model.predict(sample_train)
        label_test_predicted_ridge = ridge_model.predict(sample_test)
        
        label_train_predicted_standard = standard_model.predict(sample_train)
        label_test_predicted_standard = standard_model.predict(sample_test)
        
        # Calculate the error in each prediction
        error_train_ridge[k,i] = 1 - (accuracy_score(label_train, label_train_predicted_ridge))
        error_test_ridge[k,i] = 1 - (accuracy_score(label_test, label_test_predicted_ridge))
        
        error_train_standard[k,i] = 1 - (accuracy_score(label_train, label_train_predicted_standard))
        error_test_standard[k,i] = 1 - (accuracy_score(label_test, label_test_predicted_standard))
        
        # Count dropout      
        probability_difference_test1[k,i] = (abs(num_men_dropout/num_men - num_women_dropout/num_women))
        probability_difference_test2[k,i] = (abs(num_urban_dropout/num_urban - num_rural_dropout/num_rural))
        
        # Calculate the probability we predict someone will dropout between groups (Demographic Parity)
        num_men_predict_ridge = len([e for e in range(0, num_test) if label_test_predicted_ridge[e] == 0 and sample_test[e, 0] == 0])
        num_women_predict_ridge = len([e for e in range(0, num_test) if label_test_predicted_ridge[e] == 0 and sample_test[e, 0] == 1])
        num_urban_predict_ridge = len([e for e in range(0, num_test) if label_test_predicted_ridge[e] == 0 and sample_test[e, 16] == 1])
        num_rural_predict_ridge = len([e for e in range(0, num_test) if label_test_predicted_ridge[e] == 0 and sample_test[e, 16] == 0])
        
        num_men_predict_standard = len([e for e in range(0, num_test) if label_test_predicted_standard[e] == 0 and sample_test[e, 0] == 0])
        num_women_predict_standard = len([e for e in range(0, num_test) if label_test_predicted_standard[e] == 0 and sample_test[e, 0] == 1])
        num_urban_predict_standard = len([e for e in range(0, num_test) if label_test_predicted_standard[e] == 0 and sample_test[e, 16] == 1])
        num_rural_predict_standard = len([e for e in range(0, num_test) if label_test_predicted_standard[e] == 0 and sample_test[e, 16] == 0])

        demographic_parity_test1_ridge [k,i] = (abs(num_men_predict_ridge /num_men - num_women_predict_ridge /num_women))
        demographic_parity_test2_ridge [k,i] = (abs(num_urban_predict_ridge /num_urban - num_rural_predict_ridge /num_rural))
        
        demographic_parity_test1_standard  [k,i] = (abs(num_men_predict_standard /num_men - num_women_predict_standard /num_women))
        demographic_parity_test2_standard  [k,i] = (abs(num_urban_predict_standard /num_urban - num_rural_predict_standard /num_rural))
        

# Calculate the error for each measurement
error_test_ridge_upper = error_test_ridge.max(axis = 0) - error_test_ridge.mean(axis = 0)
error_test_ridge_lower = abs(error_test_ridge.min(axis = 0) - error_test_ridge.mean(axis = 0))

error_test_standard_upper = error_test_standard.max(axis = 0) - error_test_standard.mean(axis = 0)
error_test_standard_lower = abs(error_test_standard.min(axis = 0) - error_test_standard.mean(axis = 0))

error_test_ridge_ave = error_test_ridge.mean(axis = 0)
error_test_standard_ave = error_test_standard.mean(axis = 0)

# Gender
# Average all of the runs of our test
probability_difference_test_ave1 = probability_difference_test1.mean(axis = 0)
probability_difference_test_err_upper1 = probability_difference_test1.max(axis = 0) - probability_difference_test1.mean(axis = 0)
probability_difference_test_err_lower1 = abs(probability_difference_test1.min(axis = 0) - probability_difference_test1.mean(axis = 0))

demographic_parity_test_ave1_ridge = demographic_parity_test1_ridge .mean(axis = 0)
demographic_parity_test_ave1_standard = demographic_parity_test1_standard .mean(axis = 0)

# Calculate the error for each measurement
demographic_parity_test_err_upper1_ridge = demographic_parity_test1_ridge.max(axis = 0) - demographic_parity_test1_ridge.mean(axis = 0)
demographic_parity_test_err_lower1_ridge = abs(demographic_parity_test1_ridge.min(axis = 0) - demographic_parity_test1_ridge.mean(axis = 0))
demographic_parity_test_err_upper1_standard = demographic_parity_test1_standard.max(axis = 0) - demographic_parity_test1_standard.mean(axis = 0)
demographic_parity_test_err_lower1_standard = abs(demographic_parity_test1_standard.min(axis = 0) - demographic_parity_test1_standard.mean(axis = 0))

# Address
# Average all of the runs of our test
probability_difference_test_ave2 = probability_difference_test2.mean(axis = 0)
probability_difference_test_err_upper2 = probability_difference_test2.max(axis = 0) - probability_difference_test2.mean(axis = 0)
probability_difference_test_err_lower2 = abs(probability_difference_test2.min(axis = 0) - probability_difference_test2.mean(axis = 0))

demographic_parity_test_ave2_ridge = demographic_parity_test2_ridge.mean(axis = 0)
demographic_parity_test_ave2_standard = demographic_parity_test2_standard.mean(axis = 0)

# Calculate the error for each measurement
demographic_parity_test_err_upper2_ridge = demographic_parity_test2_ridge.max(axis = 0) - demographic_parity_test2_ridge.mean(axis = 0)
demographic_parity_test_err_lower2_ridge= abs(demographic_parity_test2_ridge.min(axis = 0) - demographic_parity_test2_ridge.mean(axis = 0))
demographic_parity_test_err_upper2_standard = demographic_parity_test2_standard.max(axis = 0) - demographic_parity_test2_standard.mean(axis = 0)
demographic_parity_test_err_lower2_standard = abs(demographic_parity_test2_standard.min(axis = 0) - demographic_parity_test2_standard.mean(axis = 0))



# Plot the result
plt.figure()
plt.errorbar(x = training_instances, y = error_test_ridge_ave, yerr = [error_test_ridge_lower, error_test_ridge_upper], label='Ridge Testing Error', capsize=4, capthick = 2)
plt.errorbar(x = training_instances, y = error_test_standard_ave, yerr = [error_test_standard_lower, error_test_standard_upper], label='Standard Testing Error', capsize=4, capthick = 2)
plt.xlabel('Number of Fair Trees')
plt.ylabel('Probability')
plt.title('Random Forest with 10 Trials vs Random Forest with Ridge 10 Trials')
plt.legend()
plt.show()

# Plot the result
plt.figure()
plt.errorbar(x = training_instances, y = demographic_parity_test_ave1_standard, yerr = [demographic_parity_test_err_lower1_standard, demographic_parity_test_err_upper1_standard], label='Demographic Parity Standard', capsize=4, capthick = 2)
plt.errorbar(x = training_instances, y = demographic_parity_test_ave1_ridge, yerr = [demographic_parity_test_err_lower1_ridge, demographic_parity_test_err_upper1_ridge], label='Demographic Parity Ridge', capsize=4, capthick = 2)
plt.errorbar(x = training_instances, y = probability_difference_test_ave1, yerr = [probability_difference_test_err_lower1, probability_difference_test_err_upper1], label='Difference in True Dropout Rate', capsize=4, capthick = 2)
plt.xlabel('Number of Training Instances')
plt.ylabel('Probability')
plt.title('Random Forest with 10 Trials (Gender)')
plt.legend()
plt.show()

# Plot the result
plt.figure()
plt.errorbar(x = training_instances, y = demographic_parity_test_ave2_standard, yerr = [demographic_parity_test_err_lower2_standard, demographic_parity_test_err_upper2_standard], label='Demographic Parity Standard', capsize=4, capthick = 2)
plt.errorbar(x = training_instances, y = demographic_parity_test_ave2_ridge, yerr = [demographic_parity_test_err_lower2_ridge, demographic_parity_test_err_upper2_ridge], label='Demographic Parity Ridge', capsize=4, capthick = 2)
plt.errorbar(x = training_instances, y = probability_difference_test_ave2, yerr = [probability_difference_test_err_lower2, probability_difference_test_err_upper2], label='Difference in True Dropout Rate', capsize=4, capthick = 2)
plt.xlabel('Number of Training Instances')
plt.ylabel('Probability')
plt.title('Random Forest with 10 Trials (Address)')
plt.legend()
plt.show()
