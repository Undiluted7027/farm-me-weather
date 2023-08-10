# imports of modules
import warnings, sklearn, math, random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
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
import sys
sys.path.insert(0, '/home/pi/mausam/mausam-main/models/ppDatasets')
from preproc import set_up


def LinReg(train_x, train_y, test_x, test_y):
    model = LinearRegression()
    model.fit(train_x, train_y)
    pred1 = model.predict(test_x)
    print(test_x)
    print(np.mean(pred1 - test_y)**2)
    result = pd.DataFrame({'actual':test_y,'prediction':pred1,'diff':(test_y-pred1)})
    print("Linear Regression Model Built")
    return model, result


def PolReg(train_x, train_y, test_x, test_y):
    poly = PolynomialFeatures(degree=4)
    x_poly = poly.fit_transform(train_x)
    model = LinearRegression()
    poly.fit(x_poly,train_y)
    model.fit(x_poly,train_y)
    pred2 = model.predict(poly.fit_transform(test_x))
    print(np.mean(pred2 - test_y)**2)
    result = pd.DataFrame({'actual':test_y,'prediction':pred2,'diff':(test_y-pred2)})
    print("Polynomial Regression Model Built")
    return model, result, poly


def decTree(train_x, train_y, test_x, test_y):
    model = DecisionTreeRegressor(random_state=0)
    model.fit(train_x,train_y)
    pred3 = model.predict(test_x)
    print(np.mean((pred3-test_y)**2))
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


def testing():
    TL_RModel = joblib.load('nTLinearRegression')
    dates = []
    temperatures = []
    date = datetime.now()
    D = date.day
    H = date.hour
    M = date.month
    tp = 13 # Get Current Temperature
    for d in range(1,15):
        for h in range(1,25):
            testDat = np.array([D,M,H,8,74,0,9]).reshape(1,-1)
            prediction = TL_RModel.predict(testDat)
            #print(prediction,H,D,M)
            date+= timedelta(hours=1)
            dates.append(date)
            temperatures.append(list(prediction)[0])
    
    result = pd.DataFrame(list(zip(dates, temperatures)), columns=['Timestamp', 'Predicted Max. Temperature'])
    print(temperatures)
    # print(result)
    result.to_csv("result.csv")


def predictions():
    df_x, df_y = preproc.set_up("temp")
    df = pd.read_csv(r"/home/pi/mausam/mausam-main/models/Preprocessed Datasets/ppTemp.csv")
    weather_df_num=df[list(df.dtypes[df.dtypes!='object'].index)]
    df_y = weather_df_num.pop('temp')
    df_x = weather_df_num
    train_x, test_x, train_y, test_y = ml_setup(df_x, df_y)
    print("ML Setup Done!")
    print("Starting Model Building..")
    linModel, linResult = LinReg(train_x, train_y, test_x, test_y)
    joblib.dump(linModel , 'nTLinearRegression')
    polModel, polResult, poly = PolReg(train_x, train_y, test_x, test_y)
    joblib.dump(polModel , 'nTPolynomialRegression')
    joblib.dump(poly, "nTPoly")
    decTModel, decTResult = decTree(train_x, train_y, test_x, test_y)
    joblib.dump(decTModel, 'nTDecisionTree')
    ranForModel, ranForResult = RanForest(train_x, train_y, test_x, test_y)
    joblib.dump(ranForModel, 'nTRandomForest')
    print("Executing the Test!")
    testing()
    print(df.isnull().any())
    print("Script ran Successfully!")
    return True


def ml_setup(df_x, df_y):
    train_x,test_x,train_y,test_y = train_test_split(df_x, df_y,test_size=0.2,random_state=4)
    return train_x, test_x, train_y, test_y

predictions()