
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy  as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:


affair = pd.read_csv('affairs.csv')


# In[3]:


affair.head(10)


# In[4]:


import seaborn as sns
sns.boxplot(x="gender",y="yearsmarried",data=affair)


# In[5]:


from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
affair['gender']= le.fit_transform(affair['gender'].astype('str'))
affair['children']= le.fit_transform(affair['children'].astype('str'))


# In[6]:


affair.head(10)


# In[7]:


import seaborn as sns
sns.boxplot(x="age",y="affairs",data=affair)


# In[8]:


affair.apply(lambda x:x.mean()) 
affair.mean()


# In[9]:


affair.affairs.value_counts()


# In[10]:


affair.gender.value_counts()


# In[11]:


affair.age.value_counts()


# In[12]:


sns.countplot(x='gender',data= affair)


# In[13]:


sns.countplot(x='children', data= affair)


# In[14]:


sns.countplot(x='yearsmarried', data= affair)


# In[15]:


sns.distplot(affair['age'])


# In[16]:


sns.distplot(affair['gender'])


# In[17]:


sns.distplot(affair['yearsmarried'])


# In[18]:


sns.distplot(affair['occupation'])


# In[19]:


sns.distplot(affair['religiousness'])


# In[20]:


sns.distplot(affair['education'])


# In[21]:


affair['affairs'].hist()


# In[22]:


affair['age'].hist()


# In[23]:


affair


# In[24]:


affair.head(5)


# In[25]:


import statsmodels.formula.api as sm
logit_model = sm.logit('gender~affairs+age+yearsmarried+children+religiousness+rating',data = affair).fit()


# In[26]:


#logit_model.summary()
y_pred = logit_model.predict(affair)


# In[27]:


affair["pred_prob"] = y_pred


# In[28]:


affair["Att_val"] = np.zeros(601)


# In[29]:


affair.loc[y_pred>0.5,"Att_val"] = 1


# In[30]:


affair


# In[31]:


confusion_matrix = pd.crosstab(affair['gender'],affair.Att_val)


# In[32]:


confusion_matrix


# In[33]:


accuracy = (224+147)/(601) 
accuracy# ROC curve 



# In[34]:


from sklearn import metrics
fpr, tpr, threshold = metrics.roc_curve(affair.gender,affair.Att_val)


# In[35]:


plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")
 
roc_auc = metrics.auc(fpr, tpr) #


# In[36]:


roc_auc

