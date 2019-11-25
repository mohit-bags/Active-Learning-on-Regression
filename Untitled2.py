#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, metrics 
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import mean_squared_error
import math


# In[6]:


def calcurmse(K):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn import datasets, linear_model, metrics 
    from sklearn.linear_model import Ridge
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min
    from sklearn.metrics import mean_squared_error
    import math
    dataset= pd.read_csv('Downloads/slump_test.csv')

    Z=dataset.iloc[:,0:9]
    kmeans=KMeans(n_clusters=1).fit(Z)
    centers = np.array(kmeans.cluster_centers_)
    closest, _ = pairwise_distances_argmin_min(centers, Z)
    close=closest[0]
    S=Z.iloc[close:close+1,:]
    Z=Z.drop(close)
    Z=Z.reset_index().iloc[:,1:] 
    
    S=(S.reset_index()).iloc[:,1:]
    N=len(Z.index)
    
    for k in range(1,K):
        mx=0
        max=0
        for n in range(0,N-k+1):
            a=S.iloc[0:1,:].values
            b=Z.iloc[n:n+1,:].values
            min_dist=np.linalg.norm(a-b)
                      
            for p in range(0,k):
                a1=S.iloc[p:p+1,:].values
                dist = np.linalg.norm(a1-b)
                if dist<min_dist:
                    min_dist=dist  
               
           
            if min_dist>max :           #max dnx
                max=min_dist
                mx=n
        S=S.append(Z.iloc[mx:mx+1,:])
        Z=Z.drop(mx)
        Z=(Z.reset_index()).iloc[:,1:]
        S=(S.reset_index()).iloc[:,1:]
    R=S.iloc[:,:]
    a=np.random.random(K)
    
    for q in range(0,K):
        index_s=S.iloc[q,0]
        index_s-=1
        temp=dataset.iloc[index_s,8]
        a[q]=temp   
    a1=pd.DataFrame(a)
    #print(a)
    R['slump']=a1
    R=R.iloc[:,0:8]
    
    X_train = R
    y_train = a1 
    y_test=Z.iloc[:,8:9]
    X_test=Z.iloc[:,0:8]
    
    clf = Ridge(alpha=0.01)
    clf.fit(X_train, y_train) 
    Ridge(alpha=0.01, copy_X=True, fit_intercept=True, max_iter=None,
          normalize=True, random_state=None, solver='auto', tol=0.001)
    pred=clf.predict(X_test)
    mean = mean_squared_error(y_test, pred)
    rms = np.sqrt(mean)
    return rms

arr=[]
for K in range(8,21):
    arr=np.append(arr,calcurmse(K))
print(arr)
arrx=[]
arry=[]

for i in range(8,21):
	#plt.plot(i,arr[i-8],label = "line 1")
    arrx=np.append(arrx,i)
    arry=np.append(arry,arr[i-8])

import matplotlib.pyplot as plt 
# plotting the points 
plt.plot(arrx, arry, color='green', linestyle='dashed', linewidth = 3, 
		marker='o', markerfacecolor='blue', markersize=12) 
# naming the x axis 
plt.xlabel('x - axis') 
# naming the y axis 
plt.ylabel('y - axis') 
# giving a title to my graph 
plt.title('GSx') 
# function to show the plot 
plt.show() 


# In[ ]:





# In[ ]:




