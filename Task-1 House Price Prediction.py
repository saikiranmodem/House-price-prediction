#!/usr/bin/env python
# coding: utf-8

# **1  House Price Prediction-Task-1**

# In[ ]:


'''Use a dataset that includes information about
housing prices and features like square
footage, number of bedrooms, etc. to train a
model that can predict the price of a new
house'''


# In[ ]:


'''House price prediction is a machine learning task that involves using␣
 ↪historical data to build a model
 capable of estimating the prices of houses based on various features or␣
 ↪attributes.
 This process is essential for real estate, financial planning, and investment␣
 ↪decisions.
 Here's an explanation of the key steps involved in house price prediction:'''


# In[ ]:


'''
panda,numpy,matplotlib,seaborn,sklearn are the basic libraries used in the
 
email spam filtering

natural language tool kit used to study the data which means a mail
 
and visualized the data in the different graphical form(pictorial representation

and here we are using the linear regression to predict the price of a new house
 


# **2  Lets See**

# In[2]:


import os 
os.getcwd()


# In[3]:


import os
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use(['ggplot'])

from sklearn.preprocessing import StandardScaler,LabelEncoder

from sklearn.linear_model import LinearRegression


# In[6]:


# reading the csv file using the pandas

# df=pd.read_csv('F:\datasets\kc_house_data.csv')

df=pd.read_csv("C:\\Users\\modem\\OneDrive\\Desktop\\python sessions\\technohacks\\kc_house_data.csv")

df.head(10)


# In[9]:


# the size of data frame (rows,colunms)

num_rows = df.shape[0]
num_colunms = df.shape[1]

print("Number of rows ",format(num_rows))
print("Number of colunms ",format(num_colunms))


# In[10]:


df.head()


# In[11]:


df.tail()


# In[12]:


df.info(verbose=False)


# In[13]:


df.dtypes


# In[14]:


df.describe(include='all').T


# **3  Data Wrangling**

# In[15]:


df.drop(['id'],axis=1,inplace=True)


# In[16]:


df.head()


# In[17]:


le=LabelEncoder()
df['date']=le.fit_transform(df['date'])
df['date'].dtype


# **4   Exploratory Data Aalysis**

# In[18]:


# count the number of houses with unique floor values
df['floors'].value_counts().to_frame()


# In[19]:


df.hist(bins=50,figsize=(15,15))
plt.show()


# In[21]:


# determine whether houses with a waterfront view or without a waterfront view 
   # have more price outliers.
sns .boxplot(data=df,x=df['waterfront'],y=df['price'])


# In[22]:


# determine if the feature sqft_above is negatively or positively correlated
# with price.

sns.regplot(data=df,x=df['sqft_above'],y=df['price'])


# In[23]:


sns.boxplot(data=df,x=df['sqft_basement'],y=df['price'])


# In[24]:


sns.barplot(data=df,x=df['floors'],y=df['price'])


# In[25]:


sns.histplot(data=df, x=df['grade'],y=df['price'] )


# In[26]:


sns.barplot(data=df,x=df['grade'],y=df['price'])


# In[35]:


corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=0.5,
                 fmt=".2f",
                 cmap="YlGnBu");
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top- 0.5)


# In[ ]:


df.drop('price',axis=1).corrwith(df.price).
 ↪plot(kind='bar',grid=True,figsize=(10,6),title="Correlation with price ")


# In[ ]:


df.skew()


# **5  Splitting the data set**

# In[ ]:


# to use the linear regression we need to split the given data in x and y format
x=np.array(df.drop(columns='price'))
y=np.array(df.drop(columns='price'))
space=df['sqft_living']
price=df['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0,
                                               25,random_state=44)


print(f"the shape of x_train is : {x_train.shape}")
print(f'the shape of x_test is : {x_test.shape}')
print(f'the shape of y_tain is : {y_train.shape}')
print(f'the shape of y_test is {y_test.shape}')


# In[ ]:


#using the linear regression model
model3=LinearRegression()
model3.fit(x_train,y_train)
y_pred3=model3.predict(x_test)

print(f'R2 Score is : {r2_score(y_test,y_pred3)}')
print(f'Mae is : {mean_absolute_error(y_test,y_pred3)}')


# In[43]:


x_train = [0.00,0.25,0.50,0.75,1.00,1.25,1.50]
y_train = [0.00,0.25,0.50,0.75,1.00,1.25,1.50]

plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, y_train, color='blue')
plt.title("Visualization")
plt.xlabel('Space')
plt.ylabel('Price')

plt.show()


# In[45]:


plt.scatter(x_train, y_train, label='Actual data',color='blue')
plt.plot(x_train, y_train, color='red')
plt.title("visualization")
plt.xlabel('space')
plt.ylabel('price')

plt.show()


# In[ ]:




