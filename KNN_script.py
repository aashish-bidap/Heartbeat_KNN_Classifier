#!/usr/bin/env python
# coding: utf-8

# **Author** : Ashishkumar Bidap <br>
# **Course** : Predictive Analytics
# 

# **Assignment** :  Week 1-Time series classification with K-nearest neighbors (KNN).<br>
# **Goal** :  Classification accuracy obtained with KNN using four instantiations of the Minkowski distance.<br>
# 
# P = 0.5, 1, 2, and 4 <br>
# 
# K = 3, 5, 11

# In[114]:


from decimal import Decimal 
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns
from matplotlib import pyplot as plt

class KNN_Clustering:
    
    def __init__(self):
        pass
    
    def file_input(self,filename):
        dataframe = pd.read_csv(filename,delimiter="   ",header=None,engine='python')
        dataframe.columns = ['Feature_' + str(col) if col != 0 else 'Target' for col in dataframe.columns]
        return dataframe
        
    def data_preparation(self,training_dataset,testing_dataset):
        
        train_x = training_dataset.loc[:,~training_dataset.columns.isin(['Target'])]
        train_y = training_dataset.loc[:,training_dataset.columns.isin(['Target'])]
        test_x = testing_dataset.loc[:,~testing_dataset.columns.isin(['Target'])]
        test_y = testing_dataset.loc[:,testing_dataset.columns.isin(['Target'])]
        
        return train_x,train_y,test_x,test_y
    
    def p_root(self,value, root): 
        root_value = 1 / float(root) 
        return round (Decimal(value) **Decimal(root_value),3) 
    
    def minkowski_distance(self,x, y, p_value): 
        if p_value >= 1:
            return (self.p_root(sum(pow(abs(a-b), p_value) for a, b in zip(x, y)), p_value)) 
        elif (p_value < 1 and p_value > 0):
            return (self.p_root(sum(abs(a-b) for a, b in zip(x, y)), p_value)) 
    
    def most_frequent(self,List): 
        if List:
            return max(set(List), key = List.count) 
        
    def eda_data(self,dataframe):
        sns.countplot(x='Target',data=dataframe).set_title('Frequency Distribution by Target Variable')
        plt.show() 
        
    def knn_predict(self,k,p,test_x,train_x,train_y):
        pred=[]
        list_1=[]
        for i in (zip(*[test_x[col] for col in test_x])):
            cnt=0
            list_1=[]
            for j in (zip(*[train_x[col] for col in train_x])):
                list_1.append([train_y['Target'][cnt],float(self.minkowski_distance(i,j,p))])
                cnt=cnt+1
            
            list_1 = sorted(list_1,key=lambda l:l[1])
            a = [list_1[i] for i in range(0,len(list_1)) if i < k]
            a = [i[0] for i in a]      
            if len(a) != 0:
                pred.append(self.most_frequent(a))
        return pred
    
    def error_metrics(self,prediction,test_y,k,p):
        accuracy = accuracy_score(prediction,test_y)        
        error = sqrt(mean_squared_error(test_y,prediction)) 
        print('\nK = ',k)
        print('P = ',p)
        print('RMSE = ',error)
        print('Accuracy =',accuracy)
        print("")
        
        dict={'y_Actual': list(test_y['Target']),'y_Predicted':list(prediction)}
        df = pd.DataFrame(dict)
        confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
        
        print (confusion_matrix)
        print("\n----------------*****--------------------")
        


# In[115]:


knn = KNN_Clustering()


# In[116]:


training_dataset = knn.file_input('/Users/abhishekbidap/Downloads/ECG200/ECG200_TRAIN.txt')
testing_dataset  = knn.file_input('/Users/abhishekbidap/Downloads/ECG200/ECG200_TEST.txt')


# In[117]:


train_x,train_y,test_x,test_y = knn.data_preparation(training_dataset,testing_dataset)


# In[118]:


#training data
knn.eda_data(training_dataset)


# In[119]:


#testing data
knn.eda_data(testing_dataset)


# In[120]:


K_val = [3,5,11]
P_val = [0.5,1,2,4]
cnt=1
for K in K_val:
    for P in P_val: 
        print("Experiment: ",cnt)
        prediction = knn.knn_predict(K,P,test_x,train_x,train_y)
        knn.error_metrics(prediction,test_y,K,P)
        cnt=cnt+1


# In[ ]:




