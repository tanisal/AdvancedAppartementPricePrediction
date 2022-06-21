# Advanced Appartment Price Prediction Model
The project is focused on predicting the prices of appartements for several major cities in Bulgaria. In order to do so,  real data was gathered from the biggest real estate website - imot.bg , using BeautifulSoup and Selenium python libraries. The data was cleaned and transformed for better model fitting, mainly using Box-cox transformation. Some of the outliers were directly deleted. For the right transformation i used visual representation of the variabales using matplotlib and seaborn libraries. 
  The simplest way for predicting the prices of the appartements is usualy linear regression. The problem there is that almost always there will be no homoskedasticity in the residuals , which means there will be different variance in the residual errors, which violates the linear regression model robustness. Confronting this i choose to use more advanced model , stack essemble regression technique.
  
 The benefit of stacking is that it has better predicting capabilities of any other single model used. Also combining several different regression models gives us robust results. The stacking model usualy involves two levels The first one is two or more base machine learning models(level 0) and a meta-model(level 1) that combines the the predictions of the level 0 models. The most common approach to preparing the training dataset for the meta-model is via k-fold cross-validation of the base models, where the out-of-fold predictions are used as the basis for the training dataset for the meta-model.
In this project as level 0 i used:
- Ridge Regression
- Lasso REgression
- ElasticNet Regression (a combination of  Ridge and Lasso)
- Gradient Boosting Regression
- Extreme gradient boosting regression

for the meta-model i used the Extreme Boosting Regression.
  The resulted blended model we save as joblib file and later used in util.py file for loading the data needed to predict the price of the appartements via reverse proxy server.
  The project structure is compriced of design part- html,css,js files and folders and Flask server part - util.py and server.py with all its atributes needed. The project is uploaded on a EC2 AWS Cloud. Nginx server is installed on the Ubuntu platform and configured to serve as a revrse proxy server.
