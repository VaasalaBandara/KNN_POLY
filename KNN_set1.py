#!/usr/bin/env python
# coding: utf-8

# In[14]:


#classification of owners of riding mowers and non owners of riding mowers


# In[15]:


import pandas as pd


# In[16]:


#data={'owners':{'income':[60,85.5,64.8,61.5,87,110.1,108,82.8,68,93,51,81],'lotsize':[18.4,16.8,21.6,20.8,23.6,19.2,17.6,22.4,20,20.8,22,20]},'nonowners':{'income':[75,52.8,64.8,43.2,84,49.2,59.4,66,47.4,33,51,63],'lotsize':[19.6,20.8,17.2,20.4,17.6,17.6,16,18.4,16.4,18.8,14,14.8]}}


# In[17]:


data={'income_owners':[60,85.5,64.8,61.5,87,110.1,108,82.8,68,93,51,81],'lotsize_owners':[18.4,16.8,21.6,20.8,23.6,19.2,17.6,22.4,20,20.8,22,20],'income_nonowners':[75,52.8,64.8,43.2,84,49.2,59.4,66,47.4,33,51,63],'lotsize_nonowners':[19.6,20.8,17.2,20.4,17.6,17.6,16,18.4,16.4,18.8,14,14.8]}


# In[18]:


print(data)


# In[19]:


df=pd.DataFrame(data)


# In[20]:


print(df)


# In[54]:


#obtaining X and y variables
import numpy as np


# # considering income

# In[23]:


X=df.drop(['lotsize_owners','lotsize_nonowners','income_nonowners'],axis=1)


# In[24]:


print(X)


# In[25]:


y=df.drop(['lotsize_owners','income_owners','lotsize_nonowners'],axis=1)


# In[26]:


print(y)


# In[40]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5,random_state=42)


# In[41]:


print(X_train)


# In[42]:


print(X_test)


# In[43]:


print(y_train)


# In[44]:


print(y_test)


# In[45]:


features=[[X_train],[y_train]]
labels=['owners','non owners']


# In[46]:


#The features array is four dimensions
#reshaping the array to two dimensions
import numpy as np
features=np.array(features)
features=np.reshape(features,(len(features),-1))
print(features)


# In[47]:


print(labels)


# In[51]:


from sklearn.neighbors import KNeighborsClassifier
neigh=KNeighborsClassifier(n_neighbors=2)
neigh.fit(features,labels)


# In[52]:


p_array=[[X_test],[y_test]]
print(p_array)


# In[84]:


p_array=[[X_test],[y_test]]
p_array=np.array(p_array)
p_array=p_array.reshape(len(p_array),-1)
print(p_array)
pred=neigh.predict(p_array)
print(pred)


# # considering lotsize

# In[55]:


print(df)


# In[61]:


Xb=df.drop(['income_owners','income_nonowners','lotsize_nonowners'],axis=1)


# In[62]:


print(Xb)


# In[63]:


yb=df.drop(['income_owners','lotsize_owners','income_nonowners'],axis=1)


# In[64]:


print(yb)


# In[70]:


Xb_train,Xb_test,yb_train,yb_test=train_test_split(Xb,yb,test_size=0.5,random_state=42)


# In[71]:


print(Xb_train)


# In[72]:


print(Xb_test)


# In[73]:


print(yb_train)


# In[74]:


print(yb_test)


# In[77]:


features_b=[[Xb_train],[yb_train]]
labels_b=['owners','non owners']


# In[79]:


import numpy as np
features_b=np.array(features_b)
features_b=np.reshape(features_b,(len(features_b),-1))
print(features_b)


# In[80]:


print(labels_b)


# In[81]:


from sklearn.neighbors import KNeighborsClassifier
neigh=KNeighborsClassifier(n_neighbors=2)
neigh.fit(features_b,labels_b)


# In[82]:


p_array_b=[[Xb_test],[yb_test]]
print(p_array_b)


# In[85]:


p_array_b=[[Xb_test],[yb_test]]
p_array_b=np.array(p_array_b)
p_array_b=p_array.reshape(len(p_array_b),-1)
print(p_array_b)
pred=neigh.predict(p_array_b)
print(pred)


# In[ ]:




