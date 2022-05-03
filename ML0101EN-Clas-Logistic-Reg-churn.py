

get_ipython().system('pip install scikit-learn==0.23.1')




import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt





get_ipython().system('wget -O ChurnData.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv')




df = pd.read_csv("ChurnData.csv")
df.head(10)




df.info()



df_new = df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
df_new



df_new['churn'] = df_new['churn'].astype('int')




df_new.head()




X = np.asarray(df_new[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X[0:5]


# In[24]:


y = np.asarray(df_new['churn'])
y [0:5]


# Also, we normalize the dataset:
# 

# In[28]:


from sklearn import preprocessing as pre
X = pre.StandardScaler().fit(X).transform(X)
X[0:8]


# ## Train/Test dataset
# 

# We split our dataset into train and test set:
# 

# In[44]:


from sklearn.model_selection import train_test_split as f

X_train, X_test , y_train , y_test = f(X ,y , test_size = 0.2 , random_state=4)

print('Train set:' , X_train.shape, y_train.shape)
print('Test set:' , X_test.shape , y_test.shape)





from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import confusion_matrix
LR = lr(C=0.01 , solver = 'liblinear')
LR = LR.fit(X_train , y_train)
LR


# Now we can predict using our test set:
# 

# In[42]:



yhat = LR.predict(X_test)
yhat


# **predict_proba**  returns estimates for all classes, ordered by the label of classes. So, the first column is the probability of class 0, P(Y=0|X), and second column is probability of class 1, P(Y=1|X):
# 

# In[52]:


yhat_prob = LR.predict_proba(X_test)
yhat_prob


# <h2 id="evaluation">Evaluation</h2>
# 

# ### jaccard index
# 
# Let's try the jaccard index for accuracy evaluation. we can define jaccard as the size of the intersection divided by the size of the union of the two label sets. If the entire set of predicted labels for a sample strictly matches with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.
# 

# In[55]:


from sklearn.metrics import jaccard_score

jaccard_score(y_test , yhat , pos_label =0)


# ### confusion matrix
# 
# Another way of looking at the accuracy of the classifier is to look at **confusion matrix**.
# 

# In[56]:


from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))


# In[57]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')




print (classification_report(y_test, yhat))




from sklearn.metrics import log_loss
log_loss(y_test, yhat_prob)



# In[65]:


LR_new  = lr(C=0.01 , solver = 'sag').fit(X_train , y_train)
yhat_prob2  = LR_new.predict_proba(X_test)
print("loggloss" , log_loss(y_test , yhat_prob2) )


# # ~Aditya Mathur
# ## source - IBM

# In[ ]:




