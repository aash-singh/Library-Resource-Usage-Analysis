#!/usr/bin/env python
# coding: utf-8

# In[65]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[66]:


df = pd.read_csv('Data.csv')


# In[67]:


df


# In[68]:


df = df.drop(labels='Roll Number', axis=1)


# In[69]:


df


# In[70]:


df = df.replace('None', np.nan)  # Replace 'None' with NaN for consistency
df = df.replace('NONE', np.nan)


# In[71]:


df


# In[72]:


df['Average time you spend in library during weekends?(Ofc when it\'s open, give number in hours)'] =     df['Average time you spend in library during weekends?(Ofc when it\'s open, give number in hours)']     .replace({'1hr': 1.0, 'P': np.nan})


# In[73]:


df['Other than course-related books what type of books you bring to your hostel room?'].replace({'Nothing': np.nan})


# In[74]:


df


# In[75]:


df.dtypes


# In[76]:


df.shape


# In[77]:


x = df["Average time you spend in library during weekends?(Ofc when it's open, give number in hours)"].mode()[0]
df["Average time you spend in library during weekends?(Ofc when it's open, give number in hours)"].fillna(x, inplace = True)


# In[78]:


df


# In[79]:


df.drop('Other than course-related books what type of books you bring to your hostel room?', axis = 1, inplace = True)


# In[80]:


df


# In[81]:


df['What type of books you read in library?'].value_counts()


# In[82]:


df['What type of books you read in library?'].replace(np.nan,'Course-related',inplace=True)


# In[83]:


df


# In[84]:


#replacing >2 with -1 in How many times you  filled fine column
df['How many times you have given fine for not returning book before due date?'].replace('>2', '-1', inplace = True)


# In[85]:


df['For  how long you want the library to be open in a day?'].value_counts()


# In[86]:


df['For  how long you want the library to be open in a day?'].replace(np.nan, '>20hrs', inplace = True)


# In[87]:


df


# In[88]:


df['How often do you face internet issues in library?'].value_counts()


# In[89]:


df['How often do you face internet issues in library?'].replace(np.nan, 'Sometime', inplace = True)


# In[90]:


df['What do you think about internet speed in library?'].value_counts()


# In[91]:


df['What do you think about internet speed in library?'].replace(np.nan, 'Good', inplace = True)


# In[92]:


df


# In[93]:


df['Average time you spend in library during weekdays?(Give Number in hours Eg: 4, 2, 3.5, 4.5)'].value_counts()


# In[94]:


df['Average time you spend in library during weekdays?(Give Number in hours Eg: 4, 2, 3.5, 4.5)'].plot(kind = 'hist')


# In[95]:


df["Average time you spend in library during weekends?(Ofc when it's open, give number in hours)"].astype(float).value_counts()


# In[96]:


df["Average time you spend in library during weekends?(Ofc when it's open, give number in hours)"].astype(float).plot(kind = 'hist', title= 'Average time you spend in library during weekends?')


# In[97]:


frequency = df['Which source do you prefer to study from?'].value_counts()
print(frequency)


# In[98]:


print(type(frequency))


# In[99]:


mylabels = ['Online', 'Both', 'Books']
explode = (0.07, 0.07, 0)
plt.pie(frequency, labels = mylabels, autopct='%1.1f%%', explode=explode, shadow= True)
plt.show()


# In[100]:


frequency = df['Where do you prefer to study?'].value_counts()
print(frequency)


# In[101]:


explode = [0.1,0]
mylabels = ["Room", "Library"]
plt.pie(frequency, labels = mylabels, autopct='%1.1f%%', explode=explode, shadow= True)
plt.show()


# In[102]:


arr = df['What type of books you read in library?'].value_counts()


# In[103]:


mylabels = ["Course-related", "Competitive Exams", "Novels", "Newspapers", "Spirituality", "Journal"]
explode = [0.1,0,0,0,0,0]
plt.pie(arr, labels = mylabels, autopct = '%1.1f%%', explode=explode, shadow=True)
plt.show()


# In[104]:


df


# In[105]:


value = df['Are you satisfied with library timings?'].value_counts()
print(value)


# In[106]:


explode = [0.1,0]
plt.pie(value, labels = ['No','Yes'],autopct = '%1.1f%%', explode=explode, shadow= True)
plt.show()


# In[107]:


df


# In[108]:


duration = df['For  how long you want the library to be open in a day?'].value_counts()
print(duration)


# In[109]:


mylabels = ['>20hrs', '12-13hrs', '<12hrs', '14-15hrs', '16-20hrs', '>20hrs']
explode = [0.1,0,0,0,0,0]
plt.pie(duration, labels = mylabels, autopct = '%1.1f%%', explode=explode, shadow= True)
plt.show()


# In[110]:


frequency = df['How often do you face internet issues in library?'].value_counts()
print(frequency)


# In[111]:


mylabels = ['Sometime', 'Very Often', 'Rarely', 'Never']
explode = [0.1,0,0,0]
plt.pie(frequency, labels = mylabels, autopct = '%1.1f%%', explode=explode, startangle = 90, shadow=True)
plt.show()


# In[112]:


df.head()


# In[113]:


frequency = df['What do you think about internet speed in library?'].value_counts()
print(frequency)


# In[114]:


plt.pie(frequency, labels=mylabels, autopct ='%1.1f%%', explode=explode, shadow=True)
mylabels = ['Good', 'Enough', 'Not satisfactory', 'Very Good']
explode = [0.06,0.05,0,0]


# In[115]:


df.head()


# In[116]:


df['Are you satisfied with library timings?'] = df['Are you satisfied with library timings?'].replace(['Yes','No'], [1,0])


# In[117]:


df.head()


# In[118]:


#applying linear regression model to predict study hours in library  based on factors like preferred study location, satisfaction with library timings, and internet speed.


# In[119]:


df = df.drop(['How often do you face internet issues in library?', 'Timestamp','Name', 'Roll No','Email Address',"Average time you spend in library during weekends?(Ofc when it's open, give number in hours)",'What type of books you read in library?','How many times you have given fine for not returning book before due date?','For  how long you want the library to be open in a day?','Which source do you prefer to study from?'], axis=1)


# In[120]:


df.head()


# In[121]:


#Data Preprocessing
# Encode categorical variables using one-hot encoding
df = pd.get_dummies(df, columns=['Where do you prefer to study?', 'What do you think about internet speed in library?'], drop_first=True)


# In[122]:


df.head()


# In[123]:


scaler = StandardScaler()
df['Are you satisfied with library timings?'] = scaler.fit_transform(df[['Are you satisfied with library timings?']])


# In[124]:


df.head()


# In[125]:


X = df.drop("Average time you spend in library during weekdays?(Give Number in hours Eg: 4, 2, 3.5, 4.5)", axis=1)  # Features
y = df["Average time you spend in library during weekdays?(Give Number in hours Eg: 4, 2, 3.5, 4.5)"]  # Target variable


# In[126]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[132]:


df.head()

