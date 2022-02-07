#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
    


# In[51]:


data=pd.read_csv('spark_dataset_iris.csv')
data.head()
data.drop('Id',inplace=True,axis=1)


# In[52]:


data['Species'].unique()


# In[53]:


def cvt_text_to_num(obj):
    if str(obj)=='Iris-setosa':
        return 0
    elif str(obj)=='Iris-versicolor':
        return 1
    else:
        return 2


# In[54]:


data['Species']=data['Species'].apply(cvt_text_to_num)


# In[55]:


data['Species'].unique()


# In[56]:


import seaborn as sns


# In[57]:


sns.countplot(data['Species'])


# In[58]:


sns.heatmap(data.corr(),annot=True)


# In[68]:


sns.pairplot(data, hue="Species")


# In[59]:


# with this we can know all columns are closely releated to output column


# In[60]:


sns.heatmap(data.isnull())


# In[61]:


data.isnull().sum()


# In[62]:


# with above analysis no null values are there


# In[63]:


d=data.groupby('Species').mean()
d


# In[64]:


# above we can observe that mean of the columns of different types are closer


# In[70]:


from sklearn.model_selection import train_test_split


# In[71]:


x_tr,x_te,y_tr,y_te=train_test_split(data.drop('Species',axis=1),data['Species'],test_size=0.3,random_state=101)


# In[73]:


from sklearn.tree import DecisionTreeClassifier


# In[74]:


model=DecisionTreeClassifier(max_depth = 6,random_state = 0,criterion = "entropy")


# In[75]:


model.fit(x_tr,y_tr)


# In[81]:


from sklearn import tree
import matplotlib.pyplot as plt


# In[84]:


feature_names=data.columns
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (8,8), dpi=300)
tree.plot_tree(model,feature_names = feature_names,class_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],filled = True,rounded=True);


# In[85]:


predic=model.predict(x_te)


# In[90]:


dg=pd.DataFrame({"predicted":predic,"Actual":y_te})
dg


# In[100]:


(dg['predicted']==dg['Actual']).value_counts()


# In[ ]:


# by above we can say our model is working fine


# In[101]:


from sklearn.metrics import classification_report


# In[104]:


print(classification_report(y_te,predic))


# In[ ]:




