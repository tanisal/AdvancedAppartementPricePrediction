
import json
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
import matplotlib.pyplot as plt 
from sklearn.linear_model import ElasticNetCV, RidgeCV,LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingCVRegressor
from scipy import stats
from scipy.special import boxcox1p,inv_boxcox1p
from scipy.stats import skew

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

    for i in x.split():
        if i =='Тухла' or i=='ЕПК' or i=='ПК' or i=='ЕПК' or i=='ДА' or i=='НЕ'  or i=='Панел':
            df3.drop(df3[(df3['floor'] == x)].index,inplace=True)
    else:
        if x.split()[0]=="Партер":  
            return 1
        else:
            return int(x.split('-')[0])



#Import th csv file into data frame
df=pd.read_csv('imotibg_varna.csv')

#Check for null data
df.isnull().sum()
#Remove the empty rows
df1=df.dropna()

df2=df1.copy()

# Clear the quaters info
df2['location']=df1['location'].apply(clear_location)
df2.shape

#Check how many unique Quaters we have listed
df2.groupby('location')['location'].agg('count').sort_values(ascending=False)

df3=df2.copy()
#Clear the Room details
#df3['details']=df2['details'].apply(lambda x: str(x))
df3['details']=df2['details'].apply(lambda x: int(x.split(' ')[1].split('-')[0]))
df3.rename(columns={'details':'rooms'},inplace=True)
df3.groupby('floor')['floor'].agg('count')
df3.floor.unique()  

#We clean the floor data
df3.drop(df3[(df3['floor'] == 'Панел')].index,inplace=True)
df3.drop(df3[(df3['floor'] == 'ЕПК')].index,inplace=True)
df3.drop(df3[(df3['floor'] == 'Тухла')].index,inplace=True)
df3.drop(df3[(df3['floor'] == 'ЕПК, 1987 г.')].index,inplace=True)
df3.drop(df3[(df3['floor'] == 'Тухла, 2021 г.')].index,inplace=True)
df3.drop(df3[(df3['floor'] == 'Тухла, 2023 г.')].index,inplace=True)
df3.drop(df3[(df3['floor'] == 'Тухла, 2022 г.')].index,inplace=True)
df3.drop(df3[(df3['floor'] == 'Тухла, 2012 г.')].index,inplace=True)
df3.drop(df3[(df3['floor'] == 'Тухла, 2013 г.')].index,inplace=True)
df3.drop(df3[(df3['floor'] == 'Панел, 1990 г.')].index,inplace=True)
df3.drop(df3[(df3['floor'] == 'Панел, 1985 г.')].index,inplace=True)
df3.drop(df3[(df3['floor'] == 'Тухла, 2005 г.')].index,inplace=True)
df3.drop(df3[(df3['floor'] == 'Тухла, 2008 г.')].index,inplace=True)
df3.drop(df3[(df3['floor'] == 'НЕ')].index,inplace=True)
df4=df3.copy()
df4.floor.unique()

#We clean the price data 
df4['price']=df4['price'].apply(lv_eur)
df4['price']=df4['price'].apply(clear_eur)
df4['m2']=df4['m2'].apply(clear_sqrm)
df4['build']=df4['build'].apply(clear_build)
df4['floor']=df3['floor'].apply(clear_floor)
df4.isnull().sum()
df4.build.unique()

df4




bcx_target= boxcox1p(df4.price,-0.25)
# log_target = np.log1p(df4.price)
# sqrt_target=(df4.price)**0.5
# re_target = 1/(df4.price)
df4.price=bcx_target



# plt.rcParams["figure.figsize"] = 13,5
# fig,ax = plt.subplots(1,2)
# sns.distplot(df4.price, label= "Orginal Skew:{0}".format(np.round(skew(df4.price),4)), color="r", ax=ax[0], axlabel="ORGINAL")
# sns.distplot(log_target, label= "Transformed Skew:{0}".format(np.round(skew(log_target),4)), color="g", ax=ax[1], axlabel="Log TRANSFORMED")
# fig.legend()
# plt.show()



# plt.rcParams["figure.figsize"] = 13,5
# fig,ax = plt.subplots(1,2)
# sns.distplot(df4.price, label= "Orginal Skew:{0}".format(np.round(skew(df4.price),4)), color="r", ax=ax[0], axlabel="ORGINAL")
# sns.distplot(sqrt_target, label= "Transformed Skew:{0}".format(np.round(skew(sqrt_target),4)), color="g", ax=ax[1], axlabel="SQUARE TRANSFORMED")
# fig.legend()
# plt.show()


# plt.rcParams["figure.figsize"] = 13,5
# fig,ax = plt.subplots(1,2)
# sns.distplot(df4.price, label= "Orginal Skew:{0}".format(np.round(skew(df4.price),4)), color="r", ax=ax[0], axlabel="ORGINAL")
# sns.distplot(re_target, label= "Transformed Skew:{0}".format(np.round(skew(re_target),4)), color="g", ax=ax[1], axlabel="RECIPROCAL TRANSFORMED")
# fig.legend()
# plt.show()


# plt.rcParams["figure.figsize"] = 13,5
# fig,ax = plt.subplots(1,2)
# sns.distplot(df4.price, label= "Orginal Skew:{0}".format(np.round(skew(df4.price),4)), color="r", ax=ax[0], axlabel="ORGINAL")
# sns.distplot(bcx_target, label= "Transformed Skew:{0}".format(np.round(skew(bcx_target),4)), color="g", ax=ax[1], axlabel="BOX-COX TRANSFORMED")
# fig.legend()
# plt.show()




df4.drop([991],0,inplace=True)
df4.drop([72],0,inplace=True)

df4.drop(df4[(df4['m2']>150)].index,inplace=True)
df4.drop(df4[(df4['price']>200000)].index,inplace=True)

#Plot the features
fig, ax =plt.subplots()
ax.scatter(x=df4.m2,y=df4.price)
plt.xlabel('square meter',fontsize=14)
plt.ylabel('price',fontsize=14)
plt.show()


#Lets plot the variable we want to predict
sns.distplot(df4['price'],fit=stats.norm)
#Get the fitted parameters used by the function
(mu,sigma)= stats.norm.fit(df4.price)
print('\n mu = {:.2f} and sigma = {:.2f}'.format(mu,sigma))

#Plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu,sigma)],loc='best')

plt.ylabel('Frequency')
plt.xlabel('SalePrice distribution')

#Q-Q plot
fig=plt.figure()
res=stats.probplot(df4.price,plot=plt)
plt.show()


#We create dummy variables for location 
dummies= pd.get_dummies(df4.location)
df5=pd.concat([df4,dummies],axis='columns')

dummies2=pd.get_dummies(df5['build'])
df6=pd.concat([df5,dummies2],axis='columns')

df7=df6.drop(['price/m2','location','build'],axis='columns')

df7=df7.astype(float)

#--------------Train, test separation---------------

X=df7.drop(['price'],axis='columns')
y=df7.price

# #That is why we import train test split model
# X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=10)

# train = pd.concat([y_train,X_train],axis="columns")
# test = pd.concat([y_test,X_test],axis="columns")

#-------------------------------Starting with Models------------------------

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

def rmse(y_test, y):
    return np.sqrt(mean_squared_error(y_test,y))

def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, inv_boxcox1p(y,0.7), scoring="neg_mean_squared_error", cv=kfolds))
    return (rmse)

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))
lasso = make_pipeline(RobustScaler(), LassoCV(alphas=alphas2, random_state=42, cv=kfolds))
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))                                
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003,))
gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)               
lightgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       learning_rate=0.01, 
                                       n_estimators=5000,
                                       max_bin=200, 
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=7,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=7,
                                       verbose=-1,
                                       )
xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:squarederror', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)


stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgboost, lightgbm),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)                                    


# score = cv_rmse(ridge)

# print("Ridge: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# score = cv_rmse(lasso)
# print("LASSO: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# score = cv_rmse(elasticnet)
# print("elastic net: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# score = cv_rmse(svr)
# print("SVR: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# score = cv_rmse(lightgbm)
# print("lightgbm: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# score = cv_rmse(gbr)
# print("gbr: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# score = cv_rmse(xgboost)
# print("xgboost: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# score = cv_rmse(stack_gen)
# print("stacked: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


print('START Fit')

print('stack_gen')
stack_gen.fit(np.array(X), np.array(y))

# #Training set performace
# stack_train_score = stack_gen.score(X_train,y_train) # Calculating score
# print('-score: %s' % stack_train_score)

print('elasticnet')
elasticnet.fit(X, y)
# #Training set performace
# elasticnet_train_score = elasticnet.score(X_train,y_train) # Calculating score
# print('-score: %s' % elasticnet_train_score)

print('Lasso')
lasso.fit(X, y)

# # #Training set performace
# lasso_train_score = lasso.score(X_train,y_train) # Calculating score
# print('-score: %s' % lasso_train_score)

print('Ridge')
ridge.fit(X, y)
# # #Training set performace
# ridge_train_score = ridge.score(X_train,y_train) # Calculating score
# print('-score: %s' % ridge_train_score)

print('GradientBoosting')
gbr.fit(X, y)
# # #Training set performace
# gbr_train_score = gbr.score(X_train,y_train) # Calculating score
# print('-score: %s' % gbr_train_score)

print('xgboost')
xgboost.fit(X, y)

# # #Training set performace
# xgboost_train_score = xgboost.score(X_train,y_train) # Calculating score
# print('-score: %s' % xgboost_train_score)


print('lightgbm')
lightgbm.fit(X, y)
# #Training set performace
# lightgbm_train_score= lightgbm.score(X_train,y_train) # Calculating score
# print('-score: %s' % lightgbm_train_score)



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
    
    return inv_boxcox1p(stack_gen.predict([x])[0],-0.25)

estimated_price= predict_price('Център',60,2,9,"Тухла")

print(estimated_price)
print('Stacked score: %s' % stack_gen.score(np.array(X),np.array(y)))

#Export our mode
from joblib import dump
dump(stack_gen,'varna_appartament_price_model.joblib')
     

#in order for our model to work we need to have the columns data
#that is why we need to export a json file
columns={
    'data_columns':[col.lower() for col in X.columns]
}
with open('varna_columns.json','w',encoding='utf-8') as f:
    f.write(json.dumps(columns))




















































#Create new column for the price per aquare meter in euro, which we will use to remove the outliers later
df4['eur_price_square']=round(df4['price']/df4['m2'])
#Check for outliers
df4.eur_price_square.describe()


#Function to remove price per m2 outliars for every location separately
def remove_pps_outliers(df):
    df_out=pd.DataFrame()
    for key,subdf in df.groupby('location'):
        mean=np.mean(subdf.eur_price_square)
        std=np.std(subdf.eur_price_square)
        reduced_df=subdf[(subdf.eur_price_square>(mean-std)) &(subdf.eur_price_square<=(mean+std))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out


#Removing the outliars, shich are 1 standart deviation apparrt from the mean
df5 = remove_pps_outliers(df4)
df5.shape

# sns.distplot(df5['price'])

#Plot for the eur price oer square meter. which shows the normal distribution
'''
matplotlib.rcParams['figure.figsize']=(20,10)
plt.hist(df5.eur_price_square,rwidth=0.8)
plt.xlabel('Price Per Square Meter')
plt.ylabel('Count')

'''

#We remove the columns that we will not need
df6=df5.drop(['price/m2','eur_price_square'],axis='columns')

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
y=df10.price
y.head()

# We divide the dataset into test, train, where we use the train dataset for the model
# and to evaluate the model performance we use the test dataset
#That is why we import train test split model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=10)

#We create linear regression model
from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
#We train the model
lr_clf.fit(X_train.values,y_train)
#Check the score of the model , the r-value
lr_clf.score(X_test.values,y_test)


#-------------------P-values-------

X_incl_const = sm.add_constant(X_train)
model_sm =sm.OLS(y_train,X_incl_const)
results = model_sm.fit()
results.rsquared

# results.bic
# results.pvalues

pd.DataFrame({'coef':results.params,'p-values':round(results.pvalues,3)})



##-------------------------Residuals-----------
##First we check is there correlation btw predicted price and actual price
corr=round(y_train.corr(results.fittedvalues),2)
corr
plt.scatter(x=y_train,y=results.fittedvalues,c='navy',alpha=0.6)
plt.plot(y_train,y_train, color="cyan")
plt.xlabel('Actual prices $y _i$',fontsize=14)
plt.ylabel('Predicted Prices $y _i$',fontsize=14)
plt.title(f'Predicted Prices vs Actual Prices. Corr({corr})',fontsize=16)
plt.show()


##Residuals vs. Predicted values
plt.scatter(x=results.fittedvalues,y=results.resid,c='navy',alpha=0.6)

plt.xlabel('Predicted Prices $y _i$',fontsize=14)
plt.ylabel('Residuals',fontsize=14)
plt.show()

# #Distribution of the residuals - checking for normality
resid_mean = round(results.resid.mean(),3)
results.resid.skew()
# print(resid_mean)


sns.histplot(results.resid,color='navy',kde=True)
plt.title('Price model: residuals')
plt.show()


#------------------------------Cross Validation-------------






#We will cross validate
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
#We get over 90% r for the different splitted random datasets
cv= ShuffleSplit(n_splits=5, test_size=0.2 , random_state=0)
cross_val_score(LinearRegression(),X.values, y,cv=cv)

#We will try to see if there is better algorithms
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

#We write a function
def find_best_model_using_gridsearchsv(X,y):
    algos={
        'linear_regression':{
            'model': LinearRegression(),
            'params':{
                'normalize':[True,False]
            }
        },
        'lasso':{
            'model':Lasso(),
            'params':{
                'alpha':[1,2],
                'selection':['random','cyclic']

            }
        },
        'decision_tree':{
            'model': DecisionTreeRegressor(),
            'params':{
            'criterion':['mse','friedman_mse'],
            'splitter':['best','random']
            }
        }
    }
    scores=[]
    #Randomly shuffls the data sets
    cv = ShuffleSplit(n_splits= 5,test_size=0.2, random_state=0)


    for algo_name, config in algos.items():
        gs= GridSearchCV(config['model'], config['params'], cv =cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model':algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_,
        })
    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

#find_best_model_using_gridsearchsv(X,y)

#Linear regression is the best in our case

##############find_best_model_using_gridsearchsv(X,y)


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

    return lr_clf.predict([x])[0]


predict_price('Център',50,3,2,"Тухла")

# #Export our mode
# import pickle
# with open('varna_appartament_price_model.pickle','wb') as f:
#     pickle.dump(lr_clf, f)

# #in order for our model to work we need to have the columns data
# #that is why we need to export a json file
# columns={
#     'data_columns':[col.lower() for col in X.columns]
# }
# with open('varna_columns.json','w',encoding='utf-8') as f:
#     f.write(json.dumps(columns))
