#!/usr/bin/env python
# coding: utf-8

# # House Price prediction in KSA

# In[1]:


#Import Libraries
import pandas as pd
from pandas.plotting import andrews_curves
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn import metrics
from statsmodels.tools.eval_measures import mse
from xgboost import XGBRegressor
from sklearn.preprocessing import scale
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error
import math 
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from matplotlib import pyplot
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
import xgboost as xgb




from scipy import stats
import seaborn as sns




# In[2]:


#Load data
data =pd.read_csv('SA_Aqar 1.csv')


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.describe().T


# In[6]:


data.info()


# # Data Preproccessing

# # 1. Full missing value

# In[7]:


# plot the data to check if there is a null value
data.isnull().sum().plot()


# In[8]:


#full missing value
data['pool'].fillna(data['pool'].mean(),inplace=True)
data['elevator'].fillna(data['elevator'].mean(),inplace=True)


# In[9]:


data.isnull().sum()


# # 2. Handling Catogorical Variable

# In[10]:


data['city'].unique()


# In[11]:


data['front'].unique()


# In[12]:


data['district'].unique()


# In[13]:


enc = OrdinalEncoder()


# In[14]:


enc.fit_transform(data[['city']])


# In[15]:


enc.fit_transform(data[['front']])


# In[16]:


data[['city']]=enc.fit_transform(data[['city']])
data[['front']]=enc.fit_transform(data[['front']])


# In[17]:


data.head()


# In[18]:


label_encoder = LabelEncoder()


# In[19]:


data['district'] = label_encoder.fit_transform(data['district'])


# In[20]:


data.head()


# In[21]:


data['district'].value_counts()


# In[22]:


data['city'].value_counts()


# In[23]:


data['front'].value_counts()


# # Outlier Detection

# In[24]:


data['price'].skew()


# In[25]:


sns.distplot(data['price'])


# In[ ]:





# In[26]:


print(data['price'].describe())
 


# In[27]:


log_price=np.log1p(data['price'])


# In[28]:


log_price


# In[29]:


print(log_price.quantile(0.10),log_price.quantile(0.95))


# In[30]:


log_price.skew(),data.price.skew()


# In[31]:


z_score = np.abs(stats.zscore(log_price))


# In[32]:


z_score


# In[33]:


np.where(z_score>3)


# In[ ]:





# In[34]:


remove_outliers = data.loc[z_score<=3]


# In[35]:


remove_outliers


# In[36]:


print('old data', len(data))
print('new data', len(remove_outliers))
print('Ourlires' , len(data)-len(remove_outliers))


# In[37]:


data = remove_outliers.copy()


# In[38]:


data['log_price'] = log_price


# In[39]:


drop_price =data.drop(['price'], axis=1)


# In[40]:


data = data = drop_price.copy()


# In[41]:


data.head(5)


# # Balance the data

# In[42]:


# Dropping Column with 0's more than the 60%
plt.figure(figsize=(20,8))
sns.heatmap(data.isin([0]),  yticklabels = False,cbar = False, cmap = 'viridis')


# In[43]:


data = data.loc[:, data.isin([0]).mean() < .6]


# # Normalizing dataset

# In[44]:


# create a scaler object
scaler = MinMaxScaler()


# In[45]:


# fit the scaler to the data and transform the data
scaled_data = scaler.fit_transform(data)


# In[ ]:





# In[46]:


# create a new DataFrame with the scaled data
scaled_df = pd.DataFrame(scaled_data, columns=data.columns)


# In[47]:


# print the scaled data
data = scaled_df.copy()


# In[48]:


data


# In[ ]:





# In[ ]:





# # Feature selection

# In[49]:


X=data.drop(['log_price'] , axis=1 )
y=data['log_price']


# In[50]:


# Perform feature selection based on top K features
k = 15 # Select the top 15 features
selector = SelectKBest(f_regression, k=k)
selector.fit(X, y)


# In[51]:


# Get the indices of the selected features
selected_features = selector.get_support(indices=True)


# In[52]:


# Get the names of the selected features
feature_names = X.columns[selected_features]
print("Selected features: ")
print(feature_names)


# In[53]:


data


# In[ ]:





# In[ ]:





# #  Train and Test sets

# In[54]:


# Update the data with just the selected features
X_selected = X.iloc[:, selected_features]
y_selected = y


# In[55]:


X = X_selected.copy()
y = y_selected.copy()


# In[ ]:





# In[56]:


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:





# In[57]:


def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=10)).mean()
    return rmse


def evaluation(y, predictions):
    r_squared =metrics.r2_score(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = math.sqrt(mean_squared_error(y, predictions))
    mae = mean_absolute_error(y, predictions)
    return r_squared, mse, rmse, mae


# In[58]:


models = pd.DataFrame(columns=["Model","R2 Score","MSE","RMSE","MAE" ,"RMSE (Cross-Validation)"])


# # Random Forest

# In[59]:


#buliding the model
ran_reg = RandomForestRegressor(n_estimators=42, random_state=100, min_samples_leaf = 2, min_samples_split=2 , )
ran_reg.fit(X_train, y_train)
predictions = ran_reg.predict(X_test)

r_squared, mse, rmse, mae  = evaluation(y_test, predictions)
print("R2 Score:", r_squared)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
rmse_cross_val = rmse_cv(ran_reg)
print("RMSE Cross-Validation:", rmse_cross_val)
print("-"*30)

new_row = {"Model": "Random Forest", "R2 Score": r_squared , "MSE": mse, "RMSE": rmse, "MAE": mae , "RMSE (Cross-Validation)":rmse_cross_val }
models = models.append(new_row, ignore_index=True)


# In[60]:


ran_reg.predict(X.head(1))


# In[ ]:





# # XGBoost

# In[ ]:





# In[61]:


xgbr = XGBRegressor(learning_rate=0.1 , gamma=0.1, n_estimators=100 ,  subsample=0.9, max_depth=8 )
xgbr.fit(X_train, y_train)
predictions = xgbr.predict(X_test)

r_squared, mse, rmse,mae = evaluation(y_test, predictions)
print("R2 Score:", r_squared)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
rmse_cross_val = rmse_cv(xgbr)
print("RMSE Cross-Validation:", rmse_cross_val)
print("-"*30)
new_row = {"Model": "XGboost", "R2 Score": r_squared , "MSE": mse, "RMSE": rmse, "MAE": mae , "RMSE (Cross-Validation)":rmse_cross_val }

models = models.append(new_row, ignore_index=True)


# In[ ]:





# # Decision Tree

# In[62]:


Dec_reg = DecisionTreeRegressor(max_depth=7, min_samples_split=2, min_samples_leaf=2 )
Dec_reg.fit(X_train, y_train)
predictions = Dec_reg.predict(X_test)

r_squared, mse, rmse,mae = evaluation(y_test, predictions)
print("R2 Score:", r_squared)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
rmse_cross_val = rmse_cv(Dec_reg)
print("RMSE Cross-Validation:", rmse_cross_val)
print("-"*30)
new_row = {"Model": "Decision Tree", "R2 Score": r_squared , "MSE": mse, "RMSE": rmse, "MAE": mae , "RMSE (Cross-Validation)":rmse_cross_val }

models = models.append(new_row, ignore_index=True)


# In[ ]:





# In[ ]:





# In[63]:


models.sort_values(by="RMSE (Cross-Validation)")


# In[65]:


plt.figure(figsize=(12,8))
sns.barplot(x=models["Model"], y=models["RMSE (Cross-Validation)"])
plt.title("Models' RMSE Scores (Cross-Validated)", size=10)
plt.xticks(rotation=30, size=12)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




