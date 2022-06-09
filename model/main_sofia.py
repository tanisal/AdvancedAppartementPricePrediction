from IPython.display import display
import json
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.compose import TransformedTargetRegressor
from  scipy import stats
import scipy as sp
import statsmodels.api as sm
import statsmodels.stats.diagnostic as diag
from statsmodels.compat import lzip
#----------------------------Data Cleaning----------------------------------#


#Function to clean the build column
def clear_build(x):
    res=x.split(":")
    if len(res)>1:
        if  res[1].strip() =="ДА" or res[1].strip() =="НЕ":
            return 'Тухла'
        if  res[1].strip() =="ПК":
            return 'ЕПК'
        else:
            return str(res[1].strip().split(',')[0])
    else:
        return str(res)

#Function to convert the price from lv into  eur
def lv_eur(price):
    if price.split(' ')[2].lower()=='лв.':
        eur = round((int("".join(price.split(" ")[0:2])))/1.952)
    else:
        return price
    return str(eur)    

#Function to clean prices which were initialy written in qur
def clear_eur(price):
    if len(price.split())>2:
        return int("".join(price.split()[0:2]))
    else:
        return float(price)

#Function to clean the square meter column
def clear_sqrm(m2):
    return float(m2.split()[0])


#Function to clean different  quaters 
def clear_location(x):
        return str(x.split(',')[1].strip())
   
 #Function to clean the floor column  and convert it to numbers
def clear_floor(x):
    if x.split()[0]=="Партер":  
        return 1
    else:
        return int(x.split('-')[0])



#Import th csv file into data frame
df=pd.read_csv('imotibg_Sofia.csv')

#Check for null data
df.isnull().sum()
#Remove the empty rows
df1=df.dropna()

df2=df1.copy()

# Clear the quaters info
df2['location']=df1['location'].apply(clear_location)


#Check how many unique Quaters we have listed
df2.groupby('location')['location'].agg('count').sort_values(ascending=False)
df3=df2.copy()

#Clear the Room details
#df3['details']=df2['details'].apply(lambda x: str(x))
df3['details']=df2['details'].apply(lambda x: int(x.split(' ')[1].split('-')[0]))
df3.rename(columns={'details':'rooms'},inplace=True)
df3.groupby('floor')['floor'].agg('count').sort_values(ascending=False)

df3['floor'].unique()

#We clean the floor data
df3.drop(df3[(df3['floor'] == 'Тухла, 2017 г.')].index,inplace=True)
df3.drop(df3[(df3['floor'] == 'Тухла, 2022 г.')].index,inplace=True)
df3.drop(df3[(df3['floor'] == 'Тухла, 2020 г.')].index,inplace=True)
df3.drop(df3[(df3['floor'] == 'ЕПК, 1987 г.')].index,inplace=True)
df3.drop(df3[(df3['floor'] == 'НЕ')].index,inplace=True)
#df3.drop(df3[(df3['build'] == 'Лок.отопл.')].index,inplace=True)
#df3.drop(df3[(df3['floor'] == 'Тухла, 1965 г.')].index,inplace=True)
#df3.drop(df3[(df3['floor'] == 'Панел, 1990 г.')].index,inplace=True)
#df3.drop(df3[(df3['floor'] == 'Панел, 1985 г.')].index,inplace=True)
df3.drop(df3[(df3['floor'] == 'ДА')].index,inplace=True)
df4=df3.copy()
df4['floor'].unique()

#We clean the price data 
df4['price']=df4['price'].apply(lv_eur)
df4['price']=df4['price'].apply(clear_eur)
df4['m2']=df4['m2'].apply(clear_sqrm)
df4['build']=df4['build'].apply(clear_build)
df4['floor']=df3['floor'].apply(clear_floor)
df4.isnull().sum()
df4.build.unique()

df4

df4.drop(df4[(df4['build'] == 'Лок.отопл.')].index,inplace=True)
df4.drop(df4[(df4['build'] == 'Гредоред')].index,inplace=True)


df5=df4.drop(['price/m2','build'],axis='columns')



#Create new column for the price per aquare meter in euro, which we will use to remove the outliers later
df4['eur_price_square']=round(df4['price']/df4['m2'])
#Check for outliers
df4.eur_price_square.describe()
min_threshold, max_threshold = df4.eur_price_square.quantile([0.05,0.95])
df4[df4['eur_price_square']>max_threshold]
df4[df4['eur_price_square']<min_threshold]
min_threshold_m2,max_threshold_m2 = df4.m2.quantile([0.05,0.95])

df4.shape

#Clean some outliers under 5% and above 95% percentile
df5=df4[(df4.eur_price_square<max_threshold)&(df4.eur_price_square>min_threshold)&(df4['m2']<max_threshold_m2)&(df4['m2']>min_threshold_m2)]
#location_stats = df3.groupby('location')['location'].agg('count').sort_values(ascending=False)
df5.shape

df5.describe()

#df5['eur_price_square'].skew()

'''
matplotlib.rcParams['figure.figsize']=(20,10)
plt.hist(df4.eur_price_square,rwidth=0.8)
plt.xlabel('Price Per Square Meter')
plt.ylabel('Count')
'''



df4.shape

#Function to remove price per m2 outliars for every location separately
def remove_pps_outliers(df):
    df_out=pd.DataFrame()
    for key,subdf in df.groupby('location'):
        mean=np.mean(subdf.eur_price_square)
        std=np.std(subdf.eur_price_square)
        reduced_df=subdf[(subdf.eur_price_square>(mean-3*std)) &(subdf.eur_price_square<=(mean+3*std))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out


#Removing the outliars, shich are 1 standart deviation apparrt from the mean
df6 = remove_pps_outliers(df4)

df6.shape

df6.describe()

#sns.pairplot(df6)

#Plot for the eur price per square meter. which shows the normal distribution
'''
matplotlib.rcParams['figure.figsize']=(20,10)
plt.hist(df5.eur_price_square,rwidth=0.8)
plt.xlabel('Price Per Square Meter')
plt.ylabel('Count')
'''

#We remove the columns that we will not need
df6=df6.drop(['price/m2','eur_price_square'],axis='columns')


#To imlement mashine learning mdel we need to use numeric inputs, that is why we need to transform 
# locatation column using psnds dummies
dummies= pd.get_dummies(df6.location)
df7=pd.concat([df6,dummies],axis='columns')
df7.head(3)

#We drop the location column , we transformed into dummies
df8=df7.drop(['location'],axis='columns')
df8.head(5)

dummies2=pd.get_dummies(df8['build'])
df9=pd.concat([df8,dummies2],axis='columns')
df9.head(3)

df10=df9.drop(['build'],axis='columns')
df10.head(3)

#--------------------------------------Regression Model------------------------


#Depended variable is price , so we have X as independend and we nees to drop the price
X=df10.drop(['price'],axis='columns')
X.head(3)

#
y=np.log(df10.price)




# We divide the dataset into test, train, where we use the train dataset for the model
# and to evaluate the model performance we use the test dataset
#That is why we import train test split model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=10)




#We create linear regression model
from sklearn.linear_model import LinearRegression
model= LinearRegression()
#We train the model
model.fit(X_train,y_train)

#Check the score of the model , the r-value
model.score(X_test,y_test)

# #What is the mean value of the predicted value we are looking at
test_predictions = model.predict(X_test)

#----------------------------------Plots------------

# # We could visualy see it
# sns.histplot(data=df10,x='price',bins=20) 

# #We need to check for residuals 
# test_residuals = y_test - test_predictions
# sns.scatterplot(x=y_test,y=test_residuals)

# sns.displot(test_residuals,bins=25,kde=True)


#we check how our data show looks like in comparison with normali distributed residuals

# fig ,ax = plt.subplots(figsize=(6,8),dpi=100)
# _ =sp.stats.probplot(test_residuals,plot=ax)







#-------------------P-values-------

X_incl_const = sm.add_constant(X_train)
model_sm =sm.OLS(y_train,X_incl_const)
results = model_sm.fit()
results.rsquared


results.bic
results.pvalues

pd.DataFrame({'coef':results.params,'p-values':round(results.pvalues,3)})



##-------------------------Residuals-----------
##First we check is there correlation btw predicted price and actual price
# corr=round(y_train.corr(results.fittedvalues),2)
# corr
# plt.scatter(x=y_train,y=results.fittedvalues,c='navy',alpha=0.6)
# plt.plot(y_train,y_train, color="cyan")
# plt.xlabel('Actual prices $y _i$',fontsize=14)
# plt.ylabel('Predicted Prices $y _i$',fontsize=14)
# plt.show()

##Residuals vs. Predicted values
plt.scatter(x=results.fittedvalues,y=results.resid,c='navy',alpha=0.6)
plt.xlabel('Predicted Prices $y _i$',fontsize=14)
plt.ylabel('Residuals',fontsize=14)
plt.show()


#Distribution of the residuals - checking for normality
resid_mean = round(results.resid.mean(),3)
results.resid.skew()
#print(resid_mean)


sns.histplot(results.resid,color='navy')
plt.title('Price model: residuals')
plt.show()

#---------------------------Check for Heteroscedasticity-----------


# # Run the White's test
# _, pval, __, f_pval = diag.het_white(results.resid, results.model.exog)
# print(pval, f_pval)
# print('-'*100)

# # print the results of the test
# if pval > 0.05:
#     print("For the White's Test")
#     print("The p-value was {:.4}".format(pval))
#     print("We fail to reject the null hypthoesis, so there is no heterosecdasticity. \n")
    
# else:
#     print("For the White's Test")
#     print("The p-value was {:.4}".format(pval))
#     print("We reject the null hypthoesis, so there is heterosecdasticity. \n")


# # Run the Breusch-Pagan test
# _, pval, __, f_pval = diag.het_breuschpagan(results.resid, results.model.exog)
# print(pval, f_pval)
# print('-'*100)

# # print the results of the test
# if pval > 0.05:
#     print("For the Breusch-Pagan's Test")
#     print("The p-value was {:.4}".format(pval))
#     print("We fail to reject the null hypthoesis, so there is no heterosecdasticity.")

# else:
#     print("For the Breusch-Pagan's Test")
#     print("The p-value was {:.4}".format(pval))
#     print("We reject the null hypthoesis, so there is heterosecdasticity.")




#Mean Squared error

# mse= round(results.mse_resid,3)


# ######### Homoskedasticity tests


# #Breusch-Pagan Test:

# name = ['LM statistic', 'LM p-value', 'F-value', 'F  p-value']
# bp_test = diag.het_breuschpagan(resid = results.resid, 
#                exog_het = pd.DataFrame(lr_log.exog, columns = lr_log.exog_names))
# print(pd.DataFrame([np.round(bp_test, 8)], columns=name))

# #We are interested in the  LM (BP) statistic and its associated  p-value. We have that the  p-value < 0.05, so we reject the null hypothesis that the residuasl are homoskedastic. Which means that the residuals are heteroskedastic.


# #Goldfeld-Quandt Test 
# name = ['F statistic', 'p-value', 'type']

# gq_test = diag.het_goldfeldquandt(y = results.model.endog, 
#                     x = results.model.exog, alternative = "two-sided")
# print(pd.DataFrame([gq_test], columns = name))

# # The   p-value > 0.05, so we have no grounds to reject the null hypothesis and conclude that the residuals are homoskedastic.


# # White Test
# name = ['LM statistic', 'LM p-value', 'F-value', 'F  p-value']
# w_test = diag.het_white(resid = results.resid, exog = results.model.exog)
# print(pd.DataFrame([np.round(w_test, 8)], columns = name))

# ####The  p-value < 0.05, so we reject the null hypothesis and conclude that the residuals are heteroskedastic.


# # Run the White's test
# _, pval, __, f_pval = diag.het_white(results.resid, results.model.exog)
# print(pval, f_pval)
# print('-'*100)

# # print the results of the test
# if pval > 0.05:
#     print("For the White's Test")
#     print("The p-value was {:.4}".format(pval))
#     print("We fail to reject the null hypthoesis, so there is no heterosecdasticity. \n")
    
# else:
#     print("For the White's Test")
#     print("The p-value was {:.4}".format(pval))
#     print("We reject the null hypthoesis, so there is heterosecdasticity. \n")


# # Run the Breusch-Pagan test
# _, pval, __, f_pval = diag.het_breuschpagan(results.resid, results.model.exog)
# print(pval, f_pval)
# print('-'*100)

# # print the results of the test
# if pval > 0.05:
#     print("For the Breusch-Pagan's Test")
#     print("The p-value was {:.4}".format(pval))
#     print("We fail to reject the null hypthoesis, so there is no heterosecdasticity.")

# else:
#     print("For the Breusch-Pagan's Test")
#     print("The p-value was {:.4}".format(pval))
#     print("We reject the null hypthoesis, so there is heterosecdasticity.")



#----------------------WLS model, regarding the heteroskedasticity--------------------

lr_log = sm.OLS(np.log(y), sm.add_constant(X))    #
lr_log_fit= lr_log.fit()
lr_log_fit.summary()


# # Run the Breusch-Pagan test
# _, pval, __, f_pval = diag.het_breuschpagan(results.resid, results.model.exog)
# print(pval, f_pval)
# print('-'*100)

# # print the results of the test
# if pval > 0.05:
#     print("For the Breusch-Pagan's Test")
#     print("The p-value was {:.4}".format(pval))
#     print("We fail to reject the null hypthoesis, so there is no heterosecdasticity.")

# else:
#     print("For the Breusch-Pagan's Test")
#     print("The p-value was {:.4}".format(pval))
#     print("We reject the null hypthoesis, so there is heterosecdasticity.")


# ######### Homoskedasticity tests


# #Breusch-Pagan Test:

# name = ['LM statistic', 'LM p-value', 'F-value', 'F  p-value']
# bp_test = diag.het_breuschpagan(resid = results.resid, 
#                exog_het = pd.DataFrame(lr_log.exog, columns = lr_log.exog_names))
# print(pd.DataFrame([np.round(bp_test, 8)], columns=name))

# # #We are interested in the  LM (BP) statistic and its associated  p-value. We have that the  p-value < 0.05, so we reject the null hypothesis that the residuasl are homoskedastic. Which means that the residuals are heteroskedastic.


# # #Goldfeld-Quandt Test 
# name = ['F statistic', 'p-value', 'type']

# gq_test = diag.het_goldfeldquandt(y = results.model.endog, 
#                     x = results.model.exog, alternative = "two-sided")
# print(pd.DataFrame([gq_test], columns = name))

# # # The   p-value > 0.05, so we have no grounds to reject the null hypothesis and conclude that the residuals are homoskedastic.


# # # White Test
# name = ['LM statistic', 'LM p-value', 'F-value', 'F  p-value']
# w_test = diag.het_white(resid = results.resid, exog = results.model.exog)
# print(pd.DataFrame([np.round(w_test, 8)], columns = name))

# # ####The  p-value < 0.05, so we reject the null hypothesis and conclude that the residuals are heteroskedastic.


# # Run the White's test
# _, pval, __, f_pval = diag.het_white(results.resid, results.model.exog)
# print(pval, f_pval)
# print('-'*100)

# # print the results of the test
# if pval > 0.05:
#     print("For the White's Test")
#     print("The p-value was {:.4}".format(pval))
#     print("We fail to reject the null hypthoesis, so there is no heterosecdasticity. \n")
    
# else:
#     print("For the White's Test")
#     print("The p-value was {:.4}".format(pval))
#     print("We reject the null hypthoesis, so there is heterosecdasticity. \n")


# # Run the Breusch-Pagan test
# _, pval, __, f_pval = diag.het_breuschpagan(results.resid, results.model.exog)
# print(pval, f_pval)
# print('-'*100)

# # print the results of the test
# if pval > 0.05:
#     print("For the Breusch-Pagan's Test")
#     print("The p-value was {:.4}".format(pval))
#     print("We fail to reject the null hypthoesis, so there is no heterosecdasticity.")

# else:
#     print("For the Breusch-Pagan's Test")
#     print("The p-value was {:.4}".format(pval))
#     print("We reject the null hypthoesis, so there is heterosecdasticity.")











####### Ploting the residuals Q-Q plot

fig = plt.figure(num = 2, figsize = (10, 8))
# Plot fitted vs residual plots:
ax = fig.add_subplot(2, 2, 1)
_ = ax.plot(results.fittedvalues, results.resid, linestyle = "None", marker = "o", markeredgecolor = "black")
# Plot the residual histogram
ax = fig.add_subplot(2, 2, 2)
_ = ax.hist(results.resid, bins = 30, edgecolor = "black")
# Plot the residual Q-Q plot:
ax = fig.add_subplot(2, 1, 2)
_ = stats.probplot(results.resid, dist = "norm", plot = ax)
# Fix layout in case the labels do overlap:
_ = plt.tight_layout()
plt.show()

#Test for serial correlation
name = ['LM-stat', 'LM: p-value', 'F-value', 'F:  p-value']
bg_t = diag.acorr_breusch_godfrey(results, nlags = 2)
print(pd.DataFrame(bg_t, index = name).T)

# # p > 0.05 conclude that the WLS residuals are not serially correlated (as was the case with the OLS residuals).

BP_t = diag.het_breuschpagan(resid = e_star, exog_het = lr_log.exog)
print(pd.DataFrame(lzip(['LM statistic', 'p-value',  'F-value', 'F: p-value'], BP_t)))

# ###  The  p-value is larger - we would not reject the null hypothesis that the residuals are homoskedastic. So, our WLS procedure did take into account all of the heteroskedasticity.

# #### On the other hand, if there (hypothetically speaking) were still some heteroskedasticity - we would need to correct our WLS standard errors. We can do this quite easily with:






# Run the White's test
_, pval, __, f_pval = diag.het_white(lr_log_wls_fit.resid, lr_log_wls_fit.model.exog)
print(pval, f_pval)
print('-'*100)

# print the results of the test
if pval > 0.05:
    print("For the White's Test")
    print("The p-value was {:.4}".format(pval))
    print("We fail to reject the null hypthoesis, so there is no heterosecdasticity. \n")
    
else:
    print("For the White's Test")
    print("The p-value was {:.4}".format(pval))
    print("We reject the null hypthoesis, so there is heterosecdasticity. \n")


# Run the Breusch-Pagan test
_, pval, __, f_pval = diag.het_breuschpagan(lr_log_wls_fit.resid, lr_log_wls_fit.model.exog)
print(pval, f_pval)
print('-'*100)

# print the results of the test
if pval > 0.05:
    print("For the Breusch-Pagan's Test")
    print("The p-value was {:.4}".format(pval))
    print("We fail to reject the null hypthoesis, so there is no heterosecdasticity.")

else:
    print("For the Breusch-Pagan's Test")
    print("The p-value was {:.4}".format(pval))
    print("We reject the null hypthoesis, so there is heterosecdasticity.")






# intercept = round(lr_log_wls_fit.params[0],4)
# coefficient = round(lr_log_wls_fit.params[1],4)

# mse_resid_wls = lr_log_wls_fit.mse_resid
# mse_resid_wls

# exp = round(np.exp(intercept),4)

# display(exp)


# # test for autocorrelation
# from statsmodels.stats.stattools import durbin_watson

# # calculate the lag, optional
# lag = min(10, (len(X)//5))
# print('The number of lags will be {}'.format(lag))
# print('-'*100)

# # run the Ljung-Box test for no autocorrelation of residuals

# # test_results = diag.acorr_breusch_godfrey(est, nlags = lag, store = True)
# test_results = diag.acorr_ljungbox(lr_log_wls_fit.resid, lags = lag)

# # grab the p-values and the test statistics
# ibvalue, p_val = test_results

# # print the results of the test
# if min(p_val) > 0.05:
#     print("The lowest p-value found was {:.4}".format(min(p_val)))
#     print("We fail to reject the null hypthoesis, so there is no autocorrelation.")
#     print('-'*100)
# else:
#     print("The lowest p-value found was {:.4}".format(min(p_val)))
#     print("We reject the null hypthoesis, so there is autocorrelation.")
#     print('-'*100)



# #------------------------------------Cross Validation---------------
# #We will cross validate
# from sklearn.model_selection import ShuffleSplit
# from sklearn.model_selection import cross_val_score
# #We get over 90% r for the different splitted random datasets
# cv= ShuffleSplit(n_splits=5, test_size=0.2 , random_state=0)
# cross_val_score(LinearRegression(),X.values, y,cv=cv)

# #We will try to see if there is better algorithms
# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import Lasso
# from sklearn.tree import DecisionTreeRegressor

# #We write a function
# def find_best_model_using_gridsearchsv(X,y):
#     algos={
#         'linear_regression':{
#             'model': LinearRegression(),
#             'params':{
#                 'normalize':[True,False]
#             }
#         },
#         'lasso':{
#             'model':Lasso(),
#             'params':{
#                 'alpha':[1,2],
#                 'selection':['random','cyclic']

#             }
#         },
#         'decision_tree':{
#             'model': DecisionTreeRegressor(),
#             'params':{
#             'criterion':['mse','friedman_mse'],
#             'splitter':['best','random']
#             }
#         }
#     }
#     scores=[]
#     #Randomly shuffls the data sets
#     cv = ShuffleSplit(n_splits= 5,test_size=0.2, random_state=0)


#     for algo_name, config in algos.items():
#         gs= GridSearchCV(config['model'], config['params'], cv =cv, return_train_score=False)
#         gs.fit(X,y)
#         scores.append({
#             'model':algo_name,
#             'best_score': gs.best_score_,
#             'best_params': gs.best_params_,
#         })
#     return pd.DataFrame(scores,columns=['model','best_score','best_params'])


# #find_best_model_using_gridsearchsv(X,y)

# #Linear regression is the best in our case

# #############find_best_model_using_gridsearchsv(X,y)





def predict_price(location,m2,rooms,floor,build):
    loc_index=np.where(X.columns==location)[0][0]
    build_index=np.where(X.columns==build)[0][0]
    x=np.zeros(len(X.columns))
    x[0]=m2
    x[1]=rooms
    x[2]=floor
    if loc_index >= 0:
        x[loc_index] = 1
    if build_index>= 0:
        x[build_index] =1

    return model.predict([x])[0]


# #Export our mode
# import pickle
# with open('sofia_appartament_price_model.pickle','wb') as f:
#     pickle.dump(model, f)

# #in order for our model to work we need to have the columns data
# #that is why we need to export a json file
# columns={
#     'data_columns':[col.lower() for col in X.columns]
# }
# with open('columns_sofia.json','w',encoding='utf-8') as f:
#     f.write(json.dumps(columns))
