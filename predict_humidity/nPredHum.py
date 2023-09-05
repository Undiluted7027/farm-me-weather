# imports of modules
import warnings, sklearn, math, sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/pi/mausam/mausam-main/models/ppDatasets')
from preproc import set_up


def LinReg(train_x, train_y, test_x, test_y):
    model = LinearRegression()
    model.fit(train_x, train_y)
    lin_predict = model.predict(test_x)
    np.mean((lin_predict - test_y)**2)
    result = pd.DataFrame({'actual':test_y,'prediction':lin_predict,'diff':(test_y-lin_predict)})
    print("Linear Regression Model Built")
    return model, result

def PolReg(train_x, train_y, test_x, test_y):
    poly = PolynomialFeatures(degree=4)
    x_poly = poly.fit_transform(train_x)
    model = LinearRegression()
    poly.fit(x_poly,train_y)
    model.fit(x_poly,train_y)
    pred2 = model.predict(poly.fit_transform(test_x))
    result = pd.DataFrame({'actual':test_y,'prediction':pred2,'diff':(test_y-pred2)})
    print("Polynomial Regression Model Built")
    return model, result, poly

def decTree(train_x, train_y, test_x, test_y):
    model = DecisionTreeRegressor(random_state=0)
    model.fit(train_x,train_y)
    pred3 = model.predict(test_x)
    np.mean((pred3-test_y)**2)
    result = pd.DataFrame({'actual':test_y,'prediction':pred3,'diff':(test_y-pred3)})
    print("Decision Tree Model Built")
    return model, result

def RanForest(train_x, train_y, test_x, test_y):
    model = RandomForestRegressor(max_depth=60,random_state=0,n_estimators=100)
    model.fit(train_x,train_y)
    pred4 = model.predict(test_x)
    np.mean((pred4-test_y)**2)
    result = pd.DataFrame({'actual':test_y,'prediction':pred4,'diff':(test_y-pred4)})
    print("Random Forest Model Built")
    return model, result


def predictions():
    df_x, df_y = set_up("humi")
    print("Preprocessing Done!")
    train_x, test_x, train_y, test_y = ml_setup(df_x, df_y)
    print("ML Setup Done!")
    print("Starting Model Building..")
    linModel, linResult = LinReg(train_x, train_y, test_x, test_y)
    joblib.dump(linModel , 'nHLinearRegression')
    polModel, polResult, poly = PolReg(train_x, train_y, test_x, test_y)
    joblib.dump(polModel , 'nHPolynomialRegression')
    joblib.dump(poly, "nHPoly")
    decTModel, decTResult = decTree(train_x, train_y, test_x, test_y)
    joblib.dump(decTModel, 'nHDecisionTree')
    ranForModel, ranForResult = RanForest(train_x, train_y, test_x, test_y)
    joblib.dump(ranForModel, 'nHRandomForest')
    # print(df.isnull().any())
    print("Script ran Successfully!")
    return True


def ml_setup(df_x, df_y):
    train_x,test_x,train_y,test_y = train_test_split(df_x, df_y,test_size=0.2,random_state=4)
    return train_x, test_x, train_y, test_y


predictions()