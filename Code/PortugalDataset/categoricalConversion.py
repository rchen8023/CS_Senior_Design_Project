import pandas as pd

#Read csv file
data = pd.read_csv('student-por.csv')

# Create dummy variables for 2 columns
data_Sex = pd.get_dummies(data['sex'])
data_Add = pd.get_dummies(data['address'])
data_FamSize = pd.get_dummies(data['famsize'])
data_Pstatus = pd.get_dummies(data['Pstatus'])
data_Mjob = pd.get_dummies(data['Mjob'])
data_Fjob = pd.get_dummies(data['Fjob'])
data_Reason = pd.get_dummies(data['reason'])
data_Guard = pd.get_dummies(data['guardian'])
data_Sup = pd.get_dummies(data['schoolsup'])
data_Fup = pd.get_dummies(data['famsup'])
data_Paid = pd.get_dummies(data['paid'])
data_Act = pd.get_dummies(data['activities'])
data_Nurs = pd.get_dummies(data['nursery'])
data_Higher = pd.get_dummies(data['higher'])
data_Internet = pd.get_dummies(data['internet'])
data_Romantic = pd.get_dummies(data['romantic'])


# Concat the dummy variables together
data_concat = pd.concat([data, data_Sex, data_Add, data_FamSize, data_Pstatus, data_Mjob, data_Fjob, data_Reason, data_Guard, data_Sup, data_Fup, data_Paid, data_Act, data_Nurs, data_Higher, data_Internet, data_Romantic], axis=1)
print (data_concat.head())
###

# Drop corresponding categorical columns and last dummy variable columns.
data_concat.drop(['sex', 'M', 'address', 'R', 'famsize', 'LE3', 'Pstatus', 'A', 'Mjob', 'Fjob', 'reason', 'guardian', 
                  'schoolsup', 'no', 'famsup', 'no', 'paid', 'no', 'activities', 'nursery', 'no','higher','no','internet','no','romantic','no'], inplace=True, axis=1)

# Save the converted categorical variables to an excel file
data_concat.to_csv(path_or_buf = 'student-por_Cleaned.csv', index=False)