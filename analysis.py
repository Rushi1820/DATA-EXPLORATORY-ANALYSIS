#!/usr/bin/env python
# coding: utf-8

# In[191]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[192]:


data=pd.read_csv('TS Weather data August 2022.csv')


# In[193]:


data


# In[194]:


r2=data[0:559]
r2


# In[195]:


r2.shape


# In[196]:


r2.describe()


# In[197]:


r2.rename(columns={'Max Temp (°C)':'MAXTEMP'})


# In[198]:


r22=r2.rename(columns={'Min Temp (°C)':'MINTEMP'})
r22


# In[199]:


r2.isnull().sum()


# In[200]:


r22=r2['Min Wind Speed (Kmph)'].replace('~',np.nan,inplace=True)


# In[201]:


r22


# In[202]:


##sns.distplot(r2,kde=True)
##plt.title('Weather report of Adilabad district')
#plt.xticks(rotation = 90)
#plt.show();


# In[203]:


sns.histplot(data=r2["Max Temp (°C)"],kde=True)


# In[204]:


sns.scatterplot(data=r2,x='Mandal',y='Max Temp (°C)')
plt.xticks(rotation=90)
plt.show()


# In[205]:


r2.corr()


# In[206]:


#for particular two variables correlation 

r2['Max Temp (°C)'].corr(r2['Rain (mm)'])


# In[207]:


#SKEWNESS
r2.head()


# In[208]:


from scipy.stats import skew


# In[209]:


r2.drop(r2.columns[[0,1,2,3]],axis=1,inplace=True)
r2


# In[210]:


for col in r2:
    print(col)
    print(skew(r2[col]))
    
    plt.figure()
    sns.histplot(r2[col])
    plt.show()


# In[211]:


sns.heatmap(r2.corr(),annot=True)
plt.show()


# In[212]:


ss=r2["Max Temp (°C)"]=np.sqrt(r2["Max Temp (°C)"])
skew(ss)


# In[213]:


from scipy.stats import kurtosis


# In[214]:


for col in r2:
    print(col)
    print(kurtosis(r2[col]))
    
    plt.figure()
    sns.histplot(r2[col])
    plt.show()


# In[215]:


sss=r2["Max Temp (°C)"]=np.sqrt(r2["Max Temp (°C)"])
kurtosis(sss)


# In[216]:


sns.boxplot(r2["Max Temp (°C)"])


# In[217]:


print(np.where(r2["Max Temp (°C)"]<1225))


# In[218]:


from scipy import stats
z = np.abs(stats.zscore(r2['Max Temp (°C)']))
z


# In[219]:


tr=3
print(np.where(z>3))


# In[220]:


#outliers 
import warnings
warnings.filterwarnings('ignore')
plt.subplot(1,2,1)
sns.distplot(r2['Max Temp (°C)'])
plt.subplot(1,2,2)
sns.distplot(r2['Min Temp (°C)'])
plt.show()


# In[221]:


print("Highest allowed",r2['Max Temp (°C)'].mean() + 3*r2['Max Temp (°C)'].std())
print("Lowest allowed",r2['Max Temp (°C)'].mean() - 3*r2['Max Temp (°C)'].std())


# In[222]:


#finding outliers
r2[(r2['Max Temp (°C)'] > 1.260) | (r2['Max Temp (°C)'] < 1.221)]


# In[236]:


#trimming of outliers
newr2= r2[(r2['Max Temp (°C)'] < 1.260) & (r2['Max Temp (°C)'] > 1.221)]
newr2


# In[224]:


upper_limit = r2['Max Temp (°C)'].mean() + 3*r2['Max Temp (°C)'].std()
lower_limit = r2['Max Temp (°C)'].mean() - 3*r2['Max Temp (°C)'].std()


# In[225]:


r2['Max Temp (°C)'] = np.where(
    r2['Max Temp (°C)']>upper_limit,
    upper_limit,
    np.where(
        r2['Max Temp (°C)']<lower_limit,
        lower_limit,
        r2['Max Temp (°C)']
    )
)


# In[226]:


r2['Max Temp (°C)'].describe()


# In[227]:


#IQR BASED FILTERING
sns.boxplot(r2['Max Temp (°C)'])


# In[228]:


percentile25 = r2['Max Temp (°C)'].quantile(0.25)
percentile75 = r2['Max Temp (°C)'].quantile(0.75)


# In[229]:


q1,q3=np.percentile(r2['Max Temp (°C)'],[75,25])
iqr=q3-q1


# In[230]:


upl = percentile75 + 1.5 * iqr
lpl = percentile25 - 1.5 * iqr


# In[231]:


r2[r2['Max Temp (°C)']>upl]
r2[r2['Max Temp (°C)']<lpl]


# In[ ]:





# In[234]:





# In[ ]:




