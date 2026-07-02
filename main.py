import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

MODEL_FILE = 'model.pkl'
PIPELINE_FILE = 'Pipeline.pkl'

def build_pipeline(num_attribs , cat_attribs):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('std_scaler' , StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('one_hot_encoder' , OneHotEncoder(handle_unknown='ignore')) 
    ])

    final_pipeline = ColumnTransformer([
        ('num', num_pipeline,num_attribs),
        ('cat', cat_pipeline, cat_attribs)
    ])

    return final_pipeline

if not os.path.exists(MODEL_FILE):
    housing = pd.read_csv("housing_dataset2.csv")
    housing['listing_date'] = pd.to_datetime(housing['listing_date'])
    housing['year'] = housing['listing_date'].dt.year
    housing['month'] = housing['listing_date'].dt.month
    housing['day'] = housing['listing_date'].dt.day
    housing = housing.drop('listing_date', axis=1)

    housing['area_cat']= pd.cut(housing['area_sqft'], bins=[0,1000,2000,3000,4000,5000], labels=[1,2,3,4,5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index ,test_index in split.split(housing,housing['area_cat']):
        housing.loc[test_index].drop('area_cat', axis=1).to_csv('input.csv', index=False)
        strat_train_set = housing.loc[train_index].drop('area_cat', axis=1)
        

    housing= strat_train_set.copy()

    housing_labels= housing['price']
    housing_features = housing.drop('price', axis=1)

    num_attribs = housing_features.drop(['property_id', 'city','property_type', 'furnished', 'market_segment'],axis=1).columns.tolist()
    cat_attribs =  ['city','property_type', 'furnished', 'market_segment'] 

    pipeline = build_pipeline(num_attribs , cat_attribs)
    housing_prepared = pipeline.fit_transform(housing_features)

    model = RandomForestRegressor()
    model.fit(housing_prepared, housing_labels)
    random_forest_rmse = -cross_val_score(model, housing_prepared, housing_labels,scoring='neg_root_mean_squared_error', cv=10)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)

    print("MODEL TRAINEEDDDDDDDD.......!")

else :
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv('input.csv')
    tranform_input = pipeline.transform(input_data)
    predictions = model.predict(tranform_input)
    input_data['price'] = predictions

    input_data.to_csv("output.csv", index=False)
    print("INFERENCE COMPLETED.....!")

print("PROJECT DONEEEE....")     



