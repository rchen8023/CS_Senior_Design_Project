import pandas as pd

#Read csv file
data = pd.read_csv('harvardEdxCleanData.csv')

# Create dummy variables for 2 columns
data_Country = pd.get_dummies(data['final_cc_cname_DI'])
data_Loe = pd.get_dummies(data['LoE_DI'])
data_Gender = pd.get_dummies(data['gender'])
data_CourseID = pd.get_dummies(data['course_id'])

# Concat the dummy variables together
data_concat = pd.concat([data, data_CourseID, data_Country, data_Loe, data_Gender], axis=1)
print (data_concat.head())
###

# Drop corresponding categorical columns and last dummy variable columns.
data_concat.drop(['course_id', 'final_cc_cname_DI', 'Unknown/Other', 'LoE_DI', 'Less than Secondary', 'gender', 'o'], inplace=True, axis=1)

# Save the converted categorical variables to an excel file
data_concat.to_csv(path_or_buf = 'harvardEdxCategoricalCleaned.csv', index=False)