import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import pickle
from transformer import Transformer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from genetic import GA
from utils import *


df = pd.read_csv('football.csv')

train_df, test_df = train_test_split(df)

train_df = train_df.dropna()
test_df = test_df.dropna()

X_train, Y_train = train_df.drop('market_value', axis=1), train_df['market_value']

unnecessary_columns = ['name','club','age','position']
X_train.drop(unnecessary_columns, axis=1, inplace=True)

all_cols = X_train.columns
cat_cols = ['position_cat', 'region', 'nationality', 'new_foreign', 'club_id', 'big_club', 'new_signing', 'age_cat']
transform = Transformer(all_cols, cat_cols)

X_train = transform.start(X_train, True)
with open('transformer.obj', 'wb') as f:
    pickle.dump(transform, f)

# Nearest Neighbors
final_attr_weights = knn_optimize(X_train, Y_train)

knn_model = KNeighborsRegressor(15, metric='wminkowski', metric_params={'w':final_attr_weights})
knn_model.fit(X_train,Y_train)

with open('knn_best.pkl', 'wb') as f:
    pickle.dump(knn_model, f)

# Linear Regression
linearModel = LinearRegression()
linearModel.fit(X_train, Y_train)

with open('linear_best.pkl', 'wb') as f:
    pickle.dump(linearModel, f)


# Gradient Boosting
gbr_model = GradientBoostingRegressor(learning_rate=0.01)
gbr_model.fit(X_train,Y_train)

with open('gbr_best.pkl', 'wb') as f:
    pickle.dump(gbr_model, f)

# Support Vector
svm_model = SVR(kernel='linear', C = 1.0)
svm_model.fit(X_train,Y_train)

with open('svr_best.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

# Decision Trees
tree = DecisionTreeRegressor('mae')
tree.fit(X_train,Y_train)

with open('tree_best.pkl', 'wb') as f:
    pickle.dump(tree, f)


# Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, Y_train)

with open('random_forest_best.pkl', 'wb') as f:
    pickle.dump(rf, f)

# Lasso
lasso_model = Lasso(0.03)
lasso_model.fit(X_train,Y_train)

with open('lasso_best.pkl', 'wb') as f:
    pickle.dump(lasso_model, f)


# Ridge
ridge_model = Ridge(2.5)
ridge_model.fit(X_train, Y_train)

with open('ridge_best.pkl', 'wb') as f:
    pickle.dump(ridge_model, f)


# Test the models

X_test, Y_test = test_df.drop('market_value', axis=1), test_df['market_value']

X_test = transform.start(X_test)

models = {}
for filename in os.listdir():
    if filename.endswith('.pkl'):
        with open(filename, 'rb') as f:
            models[filename[:-4]] = pickle.load(f)

for model, obj in models.items():
    y_pred = obj.predict(X_test)
    score = mean_squared_error(Y_test.values, y_pred, squared=False)
    print(f'RMSE Score for {model} = {score}')