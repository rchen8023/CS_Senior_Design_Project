import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Do in Parallel
from multiprocessing import Process, Queue

from fairStatistics import statisticalParity, normedDisparate
from dfrfProcess import dfrfProcess

# Stores the maximum number of values used in our training sample
n = 1993
num_train = 1495
num_test = n - num_train

# hyperparameters
num_models = 100
alpha = 50
max_depth = 20
min_sample_leaf = 3

# protected class index in data
prot_class = 0

num_train_trial = 10
num_trial = 5

# Setup arrays
error_test_dfrf  = np.zeros((num_trial, num_train_trial))

probability_difference =np.zeros((num_trial, num_train_trial))

SP_dfrf = np.zeros((num_trial, num_train_trial))
ND_dfrf = np.zeros((num_trial, num_train_trial))

training_instances = np.zeros(num_train_trial)

# Repeat over many random trials
for k in range(0, num_trial):
    print('new trial', k)
    
    data = np.loadtxt('crimecommunity/crimecommunity' + str(k+1) + '.csv', dtype = float, delimiter=',')
    
    # Split dataset into training set and test set
    sample_train = data[0:num_train,:-1]
    sample_test = data[num_train:,:-1]
    
    label_train = data[0:num_train, -1]  
    label_test = data[num_train:, -1]
    
    # Remove protected features
    sample_train_removed = np.delete(sample_train, [prot_class], axis=1)
    sample_test_removed = np.delete(sample_test, [prot_class], axis=1)
    
    processes = []
    queues = []

    # Increase the number of base models
    for j in range(0, num_train_trial):
        print('increase rho', j)
        rho = .01 * j + .01
        training_instances[j] = rho
    
        q = Queue()
        p = Process(target=dfrfProcess, args=(q, num_models, max_depth, min_sample_leaf, j, rho, alpha, prot_class, sample_train, sample_train_removed, label_train))
        processes.append(p)
        queues.append(q)
        p.start()

    # Add the constructed decision trees to a list
    for j in range(0, num_train_trial):
        i = queues[j].get()
        dfrf = queues[j].get()

        # Get predicted values from values
        label_test_predicted_dfrf = dfrf.predict(sample_test)
        
        # Calculate the error in each prediction
        error_test_dfrf[k,i] = 1 - accuracy_score(label_test, label_test_predicted_dfrf)

        # Count dropout      
        probability_difference[k,i] = statisticalParity(prot_class, sample_test, label_test)
        
        SP_dfrf[k,i] = statisticalParity(prot_class, sample_test, label_test_predicted_dfrf)
        ND_dfrf[k,i] = normedDisparate(prot_class, sample_test, label_test_predicted_dfrf)
 
        processes[i].join()       

# Calculate the error for each measurement
error_test_dfrf_ave = error_test_dfrf.mean(axis = 0)

# Average all of the runs of our test
probability_difference = probability_difference.mean(axis = 0)
SP_dfrf = SP_dfrf.mean(axis = 0)
ND_dfrf = ND_dfrf.mean(axis = 0)

# Plot the result
plt.figure()
plt.errorbar(x = training_instances, y = error_test_dfrf_ave, label='dfrf', capsize=4, capthick = 2, color='blue')
plt.xlabel('Rho')
plt.ylabel('Error')
plt.title('Community Test Error 10 Trials (Protecting Race)')
plt.legend()
plt.show()

plt.figure()
plt.errorbar(x = training_instances, y = SP_dfrf, label='dfrf', capsize=4, capthick = 2, color='blue')
plt.errorbar(x = training_instances, y = probability_difference, label='Difference in True Rate', capsize=4, capthick = 2, color='green')
plt.xlabel('Rho')
plt.ylabel('Statistical Parity')
plt.title('Community Test Statistical Parity 10 Trials (Race)')
plt.legend()
plt.show()

plt.figure()
plt.errorbar(x = training_instances, y = ND_dfrf, label='dfrf', capsize=4, capthick = 2, color='blue')
plt.xlabel('Rho')
plt.ylabel('Normed Disparate')
plt.title('Community Test Normed Disparate 10 Trials (Race)')
plt.legend()
plt.show()

plt.figure()
plt.errorbar(x = training_instances, y = error_test_dfrf_ave, label='dfrf', capsize=4, capthick = 2, color='blue')
plt.xlabel('Rho')
plt.ylabel('Error')
plt.legend()
plt.show()

fig, ax1 = plt.subplots()
t = training_instances

color = 'tab:red'
ax1.set_xlabel('Rho')
ax1.set_ylabel('Error', color=color)
ax1.plot(t, error_test_dfrf_ave, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Statistical Parity', color=color)  # we already handled the x-label with ax1
ax2.plot(t, SP_dfrf, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
