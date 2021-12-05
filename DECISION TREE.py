#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[4]:


df=pd.read_csv(r"C:\Users\pc\Downloads\diabetes.csv")


# In[5]:


df


# In[6]:


df.columns


# In[7]:


I=df.drop(['Outcome'],axis=1)


# In[8]:


I


# In[9]:


D=df.drop(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'],axis=1)


# In[10]:


D


# In[41]:


cls()


# In[40]:


cls()


# In[38]:


from sklearn.model_selection import train_test_split
I_train,I_test,D_train,D_test=train_test_split(I,D,test_size=0.25,random_state=0)


# In[42]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(I_train,D_train)


# In[44]:


p=dt.predict(I_test)


# In[45]:


p


# In[47]:


D_test.flatten()


# In[60]:


com=pd.DataFrame([{'ACTUAL':D_test.flatten()},{"PREDICTED":p}])


# In[61]:


com


# In[55]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[56]:


confusion_matrix(D_test,p)


# In[57]:


accuracy_score(D_test,p)


# In[74]:


from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




