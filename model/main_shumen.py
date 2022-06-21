import json
import pandas as pd
import numpy as np
import seaborn as sns
import warnings


def ignore_warn(*args, **kwargs):
    pass


warnings.warn = ignore_warn  # ignore annoying warning (from sklearn and seaborn)
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV, RidgeCV, LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingCVRegressor
from scipy import stats
from scipy.special import boxcox1p, inv_boxcox1p
from scipy.stats import skew


# ----------------------------Data Cleaning----------------------------------#


def clear_build(x):
    """Function to clean the build column"""
    res = x.split(":")
    if len(res) > 1:
        if res[1].strip() == "ДА" or res[1].strip() == "НЕ":
            return "Тухла"
        if res[1].strip() == "ПК":
            return "ЕПК"
        else:
            return str(res[1].strip().split(",")[0])
    else:
        return str(res)


def lv_eur(price):
    """Function to convert the price from lv into  eur"""
    if price.split(" ")[2].lower() == "лв.":
        eur = round((int("".join(price.split(" ")[0:2]))) / 1.952)
    else:
        return price
    return str(eur)


def clear_eur(price):
    """Function to clean prices which were initialy written in qur"""
    if len(price.split()) > 2:
        return int("".join(price.split()[0:2]))
    else:
        return float(price)


def clear_sqrm(m2):
    """Function to clean the square meter column"""
    return float(m2.split()[0])


# Function to clean different  quaters
def clear_location(x):
    return str(x.split(",")[1].strip())


def clear_floor(x):
    """Function to clean the floor column  and convert it to numbers"""
    if x.split()[0] == "Партер":
        return 1
    else:
        return int(x.split("-")[0])


# ---------------------------------------------Data cleaning-------------------------
# Import th csv file into data frame
df = pd.read_csv("imotibg_shumen.csv")

# Check for null data
df.isnull().sum()
# Remove the empty rows
df1 = df.dropna()

df2 = df1.copy()

# Clear the quaters info
df2["location"] = df1["location"].apply(clear_location)


# Check how many unique Quaters we have listed
df2.groupby("location")["location"].agg("count").sort_values(ascending=False)

df3 = df2.copy()
# Clear the Room details
# df3['details']=df2['details'].apply(lambda x: str(x))
df3["details"] = df2["details"].apply(lambda x: int(x.split(" ")[1].split("-")[0]))
df3.rename(columns={"details": "rooms"}, inplace=True)
df3.groupby("floor")["floor"].agg("count")
# We clean the floor data
df3.drop(df3[(df3["floor"] == "Панел")].index, inplace=True)
df3.drop(df3[(df3["floor"] == "ЕПК")].index, inplace=True)
df3.drop(df3[(df3["floor"] == "Тухла")].index, inplace=True)
df3.drop(df3[(df3["floor"] == "ЕПК, 1987 г.")].index, inplace=True)
df3.drop(df3[(df3["floor"] == "Тухла, 2021 г.")].index, inplace=True)
df3.drop(df3[(df3["floor"] == "Тухла, 2008 г.")].index, inplace=True)
df3.drop(df3[(df3["floor"] == "Тухла, 1965 г.")].index, inplace=True)
df3.drop(df3[(df3["floor"] == "Панел, 1990 г.")].index, inplace=True)
df3.drop(df3[(df3["floor"] == "Панел, 1985 г.")].index, inplace=True)
df3.drop(df3[(df3["floor"] == "ДА")].index, inplace=True)
df4 = df3.copy()
# We clean the price data
df4["price"] = df4["price"].apply(lv_eur)
df4["price"] = df4["price"].apply(clear_eur)
df4["m2"] = df4["m2"].apply(clear_sqrm)
df4["build"] = df4["build"].apply(clear_build)
df4["floor"] = df3["floor"].apply(clear_floor)
df4.isnull().sum()
df4.shape

# Plot the features
fig, ax = plt.subplots()
ax.scatter(x=df4.m2, y=df4.price)
plt.xlabel("square meter", fontsize=14)
plt.ylabel("price", fontsize=14)
plt.show()

# Remove some outliers
df4.drop(df4[(df4["m2"] > 125)].index, inplace=True)
df4.drop(df4[(df4["price"] > 110000)].index, inplace=True)
df4.drop(df4[(df4["price"] < 20000)].index, inplace=True)
df4.drop(df4[(df4["m2"] < 30)].index, inplace=True)
df5 = df4.copy()
df5.shape


# Plot the features again
fig, ax = plt.subplots()
ax.scatter(x=df5.m2, y=df5.price)
plt.xlabel("square meter", fontsize=14)
plt.ylabel("price", fontsize=14)
plt.show()

# WE create new feature, to clear the outliers
df5["e_price_square"] = round(df5["price"] / df5["m2"], 0)
df5["e_price_square"].describe()


def remove_pps_outliers(df):
    """WE create a function that clears outliers for more than one standart deviation for every single location"""

    df_out = pd.DataFrame()
    for key, subdf in df.groupby("location"):
        m = np.mean(subdf.e_price_square)
        st = np.std(subdf.e_price_square)
        reduced_df = subdf[
            (subdf.e_price_square > (m - st)) & (subdf.e_price_square <= (m + st))
        ]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out


df5 = remove_pps_outliers(df5)
df5.shape

# Plot the features
fig, ax = plt.subplots()
ax.scatter(x=df5.m2, y=df5.price)
plt.xlabel("square meter", fontsize=14)
plt.ylabel("price", fontsize=14)
plt.show()

# Lets plot the variable we want to predict
sns.distplot(df5["price"], fit=stats.norm)
# Get the fitted parameters used by the function
(mu, sigma) = stats.norm.fit(df5.price)
print("\n mu = {:.2f} and sigma = {:.2f}".format(mu, sigma))

# Plot the distribution
plt.legend(
    ["Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )".format(mu, sigma)], loc="best"
)

plt.ylabel("Frequency")
plt.xlabel("SalePrice distribution")

# Q-Q plot
fig = plt.figure()
res = stats.probplot(df5.price, plot=plt)
plt.show()


# -----------------Checking for normality approach-----------


lbda = 0.7
bcx_target = boxcox1p(df5.price, lbda)
log_target = np.log1p(df5.price)
sqrt_target = (df5.price) ** 0.5
re_target = 1 / (df5.price)


plt.rcParams["figure.figsize"] = 13, 5
fig, ax = plt.subplots(1, 2)
sns.distplot(
    df4.price,
    label="Orginal Skew:{0}".format(np.round(skew(df5.price), 4)),
    color="r",
    ax=ax[0],
    axlabel="ORGINAL",
)
sns.distplot(
    log_target,
    label="Transformed Skew:{0}".format(np.round(skew(log_target), 4)),
    color="g",
    ax=ax[1],
    axlabel="Log TRANSFORMED",
)
fig.legend()
plt.show()


plt.rcParams["figure.figsize"] = 13, 5
fig, ax = plt.subplots(1, 2)
sns.distplot(
    df4.price,
    label="Orginal Skew:{0}".format(np.round(skew(df5.price), 4)),
    color="r",
    ax=ax[0],
    axlabel="ORGINAL",
)
sns.distplot(
    sqrt_target,
    label="Transformed Skew:{0}".format(np.round(skew(sqrt_target), 4)),
    color="g",
    ax=ax[1],
    axlabel="SQUARE TRANSFORMED",
)
fig.legend()
plt.show()


plt.rcParams["figure.figsize"] = 13, 5
fig, ax = plt.subplots(1, 2)
sns.distplot(
    df4.price,
    label="Orginal Skew:{0}".format(np.round(skew(df5.price), 4)),
    color="r",
    ax=ax[0],
    axlabel="ORGINAL",
)
sns.distplot(
    re_target,
    label="Transformed Skew:{0}".format(np.round(skew(re_target), 4)),
    color="g",
    ax=ax[1],
    axlabel="RECIPROCAL TRANSFORMED",
)
fig.legend()
plt.show()


plt.rcParams["figure.figsize"] = 13, 5
fig, ax = plt.subplots(1, 2)
sns.distplot(
    df4.price,
    label="Orginal Skew:{0}".format(np.round(skew(df5.price), 4)),
    color="r",
    ax=ax[0],
    axlabel="ORGINAL",
)
sns.distplot(
    bcx_target,
    label="Transformed Skew:{0}".format(np.round(skew(bcx_target), 4)),
    color="g",
    ax=ax[1],
    axlabel="BOX-COX TRANSFORMED",
)
fig.legend()
plt.show()


# Lets plot the variable we want to predict
sns.distplot(df5["price"], fit=stats.norm)
# Get the fitted parameters used by the function
(mu, sigma) = stats.norm.fit(df4.price)
print("\n mu = {:.2f} and sigma = {:.2f}".format(mu, sigma))

# Plot the distribution
plt.legend(
    ["Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )".format(mu, sigma)], loc="best"
)

plt.ylabel("Frequency")
plt.xlabel("SalePrice distribution")

# Q-Q plot
fig = plt.figure()
res = stats.probplot(df5.price, plot=plt)
plt.show()

# We create dummy variables for location
dummies = pd.get_dummies(df5.location)
df6 = pd.concat([df5, dummies], axis="columns")

dummies2 = pd.get_dummies(df6["build"])
df7 = pd.concat([df6, dummies2], axis="columns")

df8 = df7.drop(["price/m2", "location", "build", "e_price_square"], axis="columns")

df8 = df8.astype(float)
df8


X = df8.drop(["price"], axis="columns")
y = df8.price


# -------------------------------Starting with Models------------------------
# We set k-fold cross validation , so we do not need to split out model into train, test datasets
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)


def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# Setting cross validation score
def cv_rmse(model, X=X):
    rmse = np.sqrt(
        -cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds)
    )
    return rmse


# Different alphas for the CV models
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

# Setting the Ridge regression with CV for best prefrormnce
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))
# Setting Lasso regresssion with CV
lasso = make_pipeline(
    RobustScaler(), LassoCV(alphas=alphas2, random_state=42, cv=kfolds)
)
# Same for Elastic Net Regularization
elasticnet = make_pipeline(
    RobustScaler(), ElasticNetCV(alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio)
)
# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(
    n_estimators=3000,
    learning_rate=0.05,
    max_depth=4,
    max_features="sqrt",
    min_samples_leaf=15,
    min_samples_split=10,
    loss="huber",
    random_state=42,
)
# Setting the extreme gradient boosting
xgboost = XGBRegressor(
    learning_rate=0.01,
    n_estimators=3460,
    max_depth=3,
    min_child_weight=0,
    gamma=0,
    subsample=0.7,
    colsample_bytree=0.7,
    objective="reg:squarederror",
    nthread=-1,
    scale_pos_weight=1,
    seed=27,
    reg_alpha=0.00006,
)

# Compiling the stack regressor
stack_gen = StackingCVRegressor(
    regressors=(ridge, lasso, elasticnet, gbr, xgboost),
    meta_regressor=xgboost,
    use_features_in_secondary=True,
)

# Checking the score for every single regression
score = cv_rmse(ridge)
print("Ridge: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(lasso)
print("LASSO: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(elasticnet)
print("elastic net: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


score = cv_rmse(gbr)
print("gbr: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(xgboost)
print("xgboost: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(stack_gen)
print("stacked: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# Start training  the models
print("START Fit")

print("stack_gen")
stack_model = stack_gen.fit(np.array(X), np.array(y))

print("elasticnet")
elasticnet_model = elasticnet.fit(X.values, y.values)

print("Lasso")
lasso_model = lasso.fit(X.values, y.values)

print("Ridge")
ridge_model = ridge.fit(X.values, y.values)

print("GradientBoosting")
gbr_model = gbr.fit(X.values, y.values)

print("xgboost")
xgboost_model = xgboost.fit(X.values, y.values)


# Blending the differrent models to check the rmse
def blend_models_predict(X):
    return (
        (0.05 * elasticnet_model.predict(X.values))
        + (0.1 * lasso_model.predict(X.values))
        + (0.05 * ridge_model.predict(X.values))
        + (0.2 * gbr_model.predict(X.values))
        + (0.3 * xgboost_model.predict(np.array(X)))
        + (0.3 * stack_model.predict(np.array(X)))
    )


# See the overall root mean square error on the whole data
print("RMSE score on train data:")
print(rmse(y, blend_models_predict(X)))

# Export our models, to blend them in util.py
from joblib import dump

dump(elasticnet_model, "shumen_appartament_price_model_elasticnet.joblib")
dump(lasso_model, "shumen_appartament_price_model_lasso.joblib")
dump(ridge_model, "shumen_appartament_price_model_ridge.joblib")
dump(gbr_model, "shumen_appartament_price_model_gbr.joblib")
dump(xgboost_model, "shumen_appartament_price_model_xgboost.joblib")
dump(stack_model, "shumen_appartament_price_model_stack.joblib")


# in order for our model to work we need to have the columns data
# that is why we need to export a json file
columns = {"data_columns": [col.lower() for col in X.columns]}
with open("shumen_columns.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(columns))
