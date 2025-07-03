#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd 


# In[7]:


print(np.__version__)


# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


df = pd.read_csv("creditCardFraud_28011964_120214.csv")


# In[10]:


df.head()


# In[11]:


df.columns


# In[12]:


df.info


# In[13]:


df.describe()


# In[14]:


df.describe().T


# In[19]:


df.isnull().sum()


# In[18]:


df.duplicated().sum()


# In[20]:


df.skew( )


# In[145]:


# distribution of legit transcation & fraudulent transcation
# 1 --> fraudulent transcation 
# 0--> Normal transcation
df['default payment next month'].value_counts()


# In[22]:


x = df.drop('default payment next month',axis = 1)
y = df['default payment next month']


# In[23]:


x


# In[24]:


y


# # Exploratory Data Analysis (EDA)

# In[27]:


sns.pairplot(df)
plt.show()


# In[37]:


plt.figure(figsize = (15,15))
sns.heatmap(df.corr(),annot= True)
plt.show()


# # Model Training

# In[38]:


from sklearn.model_selection import train_test_split


# In[62]:


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[63]:


X_train


# In[64]:


#scale the data 

from sklearn.preprocessing import StandardScaler


# In[65]:


train_ss= StandardScaler()
test_ss= StandardScaler()

ss_x_train = train_ss.fit_transform(X_train)
ss_x_test= test_ss.fit_transform(X_test)


# In[66]:


ss_x_train


# In[67]:


scaled_train_df = pd.DataFrame(ss_x_train, columns = X_train.columns, index= X_train.index)


# In[79]:


scaled_test_df = pd.DataFrame(ss_x_test, columns = X_test.columns, index= X_test.index)


# In[80]:


from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


# In[81]:


model_nb = GaussianNB()
model_rf = RandomForestClassifier(n_estimators=100)


# In[82]:


model_nb.fit(scaled_train_df, Y_train)
model_rf.fit(scaled_train_df, Y_train)


# # Model Evaluation

# In[85]:


#accuracy score 

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

y_pred_nb= model_nb.predict(scaled_test_df)


# In[87]:


print(f'Accuracy Score of Naive Bayes: {accuracy_score(Y_test,y_pred_nb)}')


# In[152]:


y_pred_rf =  model_rf.predict(scaled_test_df)


# In[148]:


print(f'Accuracy Score of RandomForestClassification: {accuracy_score(Y_test,y_pred_rf)}')


# In[104]:


print(f'Confusion Matrix of Naive Bayes:\n {confusion_matrix(Y_test,y_pred_nb)}')
print('\n')
print(f'Classification Report of Naive Bayes :\n {classification_report(Y_test,y_pred_nb)}')


# In[105]:


print(f'Confusion Matrix of RandomForest:\n {confusion_matrix(Y_test,y_pred_rf)}')
print('\n')
print(f'Classification Report of RandomForest :\n {classification_report(Y_test,y_pred_rf)}')


# In[109]:


from sklearn.model_selection import GridSearchCV
param_grid={'var_smoothing':[0.1,0.01,0.5,0.005,0.001,1e-6,1e-8,1e-10,1e-11]}


# In[110]:


grid = GridSearchCV(estimator = model_nb,param_grid= param_grid,cv=5,verbose = 3)


# In[111]:


grid.fit(scaled_train_df,Y_train)


# In[112]:


grid.best_params_


# In[114]:


model_nb_new =GaussianNB(var_smoothing=0.5)


# In[118]:


model_nb_new.fit(scaled_train_df, Y_train)
y_pred_nb= model_nb_new.predict(scaled_test_df)
print(f'Accuracy Score of Naive Bayes: {accuracy_score(Y_test,y_pred_nb)}')


# In[119]:


print(f'Confusion Matrix of Naive Bayes:\n {confusion_matrix(Y_test,y_pred_nb)}')
print('\n')
print(f'Classification Report of Naive Bayes :\n {classification_report(Y_test,y_pred_nb)}')


# In[121]:


param_grid = {'n_estimators':[50,100,1000,150,200],
              'max_depth': range(3,11,1),
                "random_state": [0,50,100,42],
              'criterion':['gini','entropy']
             
             }
grid2=GridSearchCV(model_rf,param_grid = param_grid, cv=5,verbose=3)
grid2.fit(scaled_train_df,Y_train)


# In[125]:


grid2.best_params_


# In[133]:


model_rf_new =RandomForestClassifier(criterion= 'entropy', max_depth= 5, n_estimators= 50, random_state= 0)


# In[149]:


model_rf_new.fit(scaled_train_df, Y_train)
y_pred_rf= model_rf_new.predict(scaled_test_df)
print(f'Accuracy Score of RandomForestClassification: {accuracy_score(Y_test,y_pred_rf)}')


# In[140]:


print(f'Confusion Matrix of RandomForest:\n {confusion_matrix(Y_test,y_pred_rf)}')
print('\n')
print(f'Classification Report of RandomForest :\n {classification_report(Y_test,y_pred_rf)}')


# In[147]:


#Alternative way 
model_rf_new =RandomForestClassifier(**grid2.best_params_)
model_rf_new.fit(scaled_train_df, Y_train)
y_pred_rf= model_rf_new.predict(scaled_test_df)
print(f'Accuracy Score of RandomForest: {accuracy_score(Y_test,y_pred_rf)}')
print(f'Confusion Matrix of RandomForest:\n {confusion_matrix(Y_test,y_pred_rf)}')
print('\n')
print(f'Classification Report of RandomForest :\n {classification_report(Y_test,y_pred_rf)}')


# # Conclusion
# This project successfully implemented two machine learning models — Naive Bayes and Random Forest — to detect fraudulent transactions.
Among the two, the Random Forest model with tuned hyperparameters provided the best accuracy of 82.17% and balance between precision and recall.

However, due to the class imbalance, the model struggled to detect fraud accurately, identifying only 4 out of 60 fraudulent transactions (Recall = 6.67%). This highlights the challenge of fraud detection, where accuracy alone is not a reliable metric.

--> 4 fraudulent transction were correctly detected
--> 56 were missed
--> Detection Rate = 7%