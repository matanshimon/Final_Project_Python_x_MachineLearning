#!/usr/bin/env python
# coding: utf-8

# # Matala model training

# ## MATAN & KARIN

# #### Import of required libraries:
# 

# In[1]:


import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from datetime import datetime
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from madlan_data_prep import prepare_data
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNetCV
import sklearn
from sklearn.compose import ColumnTransformer


# In[2]:


datafile =  "C:\\Users\קארין\\Desktop\\הנדסת תעון - קארין\\שנה ג\\סמסטר ב\\כרייה וניתוח נתונים בפייתון\\מטלות\\מטלה מסכמת\\Dataset_for_test.csv"
data = pd.read_csv(datafile)


# In[3]:


data_copy = prepare_data(data)


# In[4]:


features  = ['City', 'type', 'room_number', 'Area','floor ','price']


# In[5]:


df = data_copy[features]


# In[6]:


X = df.drop('price', axis=1)
y = df['price']


# In[7]:


# Define the column transformer for preprocessing
categorical_features = ['City', 'type']
numeric_features = ['room_number', 'Area', 'floor']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)


# In[8]:


# Split the dataset into train and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[9]:


# Preprocess the features
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)


# In[10]:


# Create an instance of the Elastic Net model
model = ElasticNet(alpha=1, l1_ratio=0.5, random_state=42)


# In[11]:


# Fit the model to the training data
model.fit(X_train_preprocessed, y_train)


# In[12]:


# Make predictions on the test data
y_pred = model.predict(X_test_preprocessed)


# In[13]:


# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)


# In[14]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse

def score_model(y_test, y_pred, model_name):
    MSE = mse(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    R_squared = r2_score(y_test, y_pred)

    print(f"Model: {model_name}, RMSE: {np.round(RMSE, 2)}, R-Squared: {np.round(R_squared, 2)}")

score_model(y_test, y_pred, "linear_regression")


# In[18]:


import pickle
import joblib


# In[22]:


# Save the model to a file
pickle.dump(model, open("trained_model.pkl", "wb"))


# In[ ]:




