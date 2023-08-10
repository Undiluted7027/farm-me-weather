import warnings, sklearn, math, sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
#from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/pi/mausam/mausam-main/models/ppDatasets')
from preproc import set_up


def logReg(train_x, test_x, train_y, test_y):
    model = LogisticRegression()
    model.fit(train_x, train_y)
    pred1 = model.predict(test_x)
    metric1 = classification_report(test_y, pred1)
    metric2 = confusion_matrix(test_y, pred1)
    metric3 = accuracy_score(test_y, pred1)
    metric4 = ((pred1-test_y)**2)
    print(np.mean(pred1 - test_y)**2)
    result = pd.DataFrame({'actual':test_y,'prediction':pred1,'diff':(test_y-pred1)})
    print("Logistic Regression Model Built")
    return model, result


def RanForest(train_x, test_x, train_y, test_y):
    model = RandomForestRegressor(max_depth=60,random_state=0,n_estimators=100)
    model.fit(train_x, train_y)
    pred2 = model.predict(test_x)
    print(np.mean(pred2 - test_y)**2)
    result = pd.DataFrame({'actual':test_y,'prediction':pred2,'diff':(test_y-pred2)})
    print("Random Forest Model Built")
    return model, result


def predictions():
    df_x, df_y = set_up("preci")
    print("Pre Processing Done!")
    train_x, test_x, train_y, test_y = ml_setup(df_x, df_y)
    print("ML Setup Done!")
    print("Starting Model Building..")
    logModel, logResult = logReg(train_x, test_x, train_y, test_y)
    joblib.dump(logModel, 'nPLogisticRegression')
    ranForModel, ranForResult = RanForest(train_x, test_x, train_y, test_y)
    joblib.dump(ranForModel, 'PRandomForest')
    print("Script ran Successfully!")
    return True


def ml_setup(df_x, df_y):
    train_x,test_x,train_y,test_y = train_test_split(df_x, df_y,test_size=0.2,random_state=4)
    return train_x, test_x, train_y, test_y

predictions()