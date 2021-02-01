import pandas as pd

#Read csv file
data = pd.read_csv('E:/regi024.csv')

# Create dummy variables for 2 columns
data_term = pd.get_dummies(data['cohort term'])
data_reg = pd.get_dummies(data['registered following term'])
data_with = pd.get_dummies(data['withdrawn?'])
data_acad = pd.get_dummies(data['academic standing'])
data_prev_reg = pd.get_dummies(data['acad standing prevents registration'])
data_gen = pd.get_dummies(data['gender'])
data_race = pd.get_dummies(data['ipeds race'])
data_ethn = pd.get_dummies(data['ethn'])
data_prim = pd.get_dummies(data['primary major'])

# Concat the dummy variables together
data_concat = pd.concat([data,data_term,data_with,data_acad,data_prev_reg,data_prim,data_gen,data_race,data_ethn,data_reg], axis=1)
print (data_concat.head())
###

# Drop corresponding categorical columns and last dummy variable columns.
data_concat.drop(['current holds','cohort', 'cohort term', 'registered following term', 'withdrawn?', 'eventually returned', 'academic standing', 'acad standing term', 'acad standing prevents registration', 'current holds', 'gender', 'ipeds race', 'ethn', 'primary major'], inplace=True, axis=1)

# Save the converted categorical variables to an excel file
data_concat.to_csv(path_or_buf = 'student_cleaned.csv', index=False)