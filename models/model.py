#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import requests
import json
import pickle
import numpy as pd
import pandas as pd

warnings.filterwarnings("ignore")
#using matplotlib.pyplot
import matplotlib.pyplot as plt
#using the seabonr library
import seaborn as sns

sns.set()
#use retina for better and more sharp and legible configurations
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[2]:


df = pd.read_csv('perovskite.csv')
df.head(30)


# In[8]:


df.describe()


# In[9]:


y=df._out_crystalscore


# In[10]:


df_features = ['_rxn_M_acid','_rxn_M_organic']


# In[11]:


X = df[df_features]


# In[12]:


X.describe()


# In[ ]:





# In[13]:


y.describe()


# In[14]:


X.head(100)


# In[15]:


y.head()


# In[16]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# In[17]:


X_train_full, X_valid_full, y_train, y_valid = train_test_split(X,y, train_size =.4, test_size = 0.6, random_state = 0)

X_training = X_train_full

X_validy = X_valid_full


# In[18]:


X_training.head(200)


# In[19]:


X_validy.head(200)


# In[27]:


y_train.head(200)


# In[20]:


y_valid.head(200)


# In[21]:


perksovites_model = DecisionTreeClassifier(random_state = 0)

perksovites_model.fit(X_training, y_train)


# In[22]:


print("Making Predictions for the following 20 reactions:")
print(X_training.head())
print("The predictionsare")
print(perksovites_model.predict(X_training.head(50)))


# In[29]:


print("Making Predictions for the following 20 reactions:")
print(X_training.head())
print("The predictionsare")
print(perksovites_model.predict(X_validy.head(50)))


# Saving the model using pickle
pickle.dump(perksovites_model, open('model.pkl','wb'))

#Loading model to compare the results
model = pickle.load( open('model.pkl', 'rb'))

# 

# 

# In[31]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[24]:


from sklearn.metrics import mean_absolute_error
# we're gonna use Mean Absolute Error to summarize the model quality


# In[25]:


# getting the predicted reaction outputs on the validation data


val_predictions = perksovites_model.predict(X_validy)
print(mean_absolute_error(y_valid, val_predictions))







