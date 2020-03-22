#!/usr/bin/env python
# coding: utf-8

# In[20]:


from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


digit = load_digits()
dir(digit)


# In[18]:


plt.gray()
for i in range(10):
    plt.matshow(digit.images[i])


# In[22]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(digit.data,digit.target,test_size=0.2)


# In[30]:


#model creation
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()


# In[32]:


reg.fit(x_train,y_train)


# In[33]:


#model accuracy
reg.score(x_test,y_test)


# In[47]:


reg.predict(digit.data[0:6])


# In[53]:


y_predict = reg.predict(x_test)
y_predict


# In[54]:


#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predict)
cm


# In[56]:


plt.figure(figsize = (10,7))
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual Truth')

