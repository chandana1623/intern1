import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

df = pd.read_csv("Titanic-Dataset.csv")

print("<====Original Data====>")
print(df)



print("<====Data after removing the duplicates====>")
df = df.drop_duplicates()
print(df)

print("<====Checking whether the data set is empty or not====>")
print(df.isnull())

print("<====Data types of all the attributes====>")
print(df.dtypes)


print("<====Filling the missing data using mean====>")
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
df['Cabin'] = df['Cabin'].fillna(df['Cabin'].mode()[0])
print(df)


le = LabelEncoder()
df['Sex_Encoded'] = le.fit_transform(df['Sex'])
df_encoded = pd.get_dummies(df,columns=['Embarked'])
print("<====After Encodeing the categorical Fearures into numerical valus====>")
print(df_encoded)



scaler = MinMaxScaler()

df[['Survived','Pclass','Age','SibSp','Parch','Fare']] = scaler.fit_transform(df[['Survived','Pclass','Age','SibSp','Parch','Fare']])
print("<====After Normalising the numerical values====>")
print(df)


df_numeric = df[['Survived','Pclass','Age','SibSp','Parch','Fare']]
plt.figure(figsize = (12,6))
plt.boxplot([df_numeric['Survived'].dropna() , df_numeric['Pclass'].dropna() , df_numeric['Age'].dropna() , df_numeric['SibSp'].dropna() , df_numeric['Parch'].dropna(), df_numeric['Fare'].dropna()] , tick_labels = ['Survived','Pclass','Age','SibSp','Parch','Fare'])
plt.title('Boxplot Before Removing Outliers')
plt.grid(True)
plt.show()



def remove_outliers(data,column):
    Q1 = data[column].quantile(0.25)
    Q2 = data[column].quantile(0.75)
    IQR = Q2 - Q1
    lower = Q1 - 1.5*IQR
    upprr = Q2 + 1.5*IQR
    
    return data[(data[column] >= lower) & (data[column] <= upprr)]

df_no_outliers = df_numeric.dropna()
df_no_outliers = remove_outliers(df_no_outliers , 'Survived')
df_no_outliers = remove_outliers(df_no_outliers , 'Pclass')
df_no_outliers = remove_outliers(df_no_outliers , 'Age')
df_no_outliers = remove_outliers(df_no_outliers , 'SibSp')
df_no_outliers = remove_outliers(df_no_outliers , 'Parch')
df_no_outliers = remove_outliers(df_no_outliers , 'Fare')

plt.figure(figsize = (12,6))

plt.boxplot([df_no_outliers['Survived'] , df_no_outliers['Pclass'] , df_no_outliers['Age'] , df_no_outliers['SibSp'] , df_no_outliers['Parch'] , df_no_outliers['Fare']] , tick_labels = ['Survived','Pclass','Age','SibSp','Parch','Fare'])

plt.title('Boxplot After Removing Outliers')
plt.grid(True)
plt.show()
Intern