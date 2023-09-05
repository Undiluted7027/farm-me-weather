import json
import warnings
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures
import pandas as ps
import numpy as np
# import matplotlib.pyplot as plt
import joblib
import plotly
import plotly.express as px
import getdata


warnings.filterwarnings('ignore')
# MODELS
MAIN_ROOT = r"models\Trained Models for Sharing\\"
TEMP = joblib.load(MAIN_ROOT+'TDecisionTree.pkl')
PREC = joblib.load(MAIN_ROOT + 'PLogisticRegression.pkl')
PPP = joblib.load(MAIN_ROOT + 'PRandomForest.pkl')
HUM = joblib.load(r'models\Predicting Humidity\nHRandomForest.pkl')

#Variables and list for data  (r stands for representation)
DR=[]
HRR=[]
TR=[]
HR=[]
PR=[]

#Calculating Maximum and minimum temperatures
MxT=[]
MnT=[]

    

key_val = {}
# Polynomial Regression
def ml():
    DATE = datetime.now()
    da = DATE.day
    hr = DATE.hour
    mn = DATE.month
    DATA_F = ()
    DAT = getdata.serial_req(port="COM7", baud_rate=9600)
    T = DAT[0]
    RH = DAT[1]
    P = DAT[2]
    # temperature, humidity and pressure from 6 hours ago
    t6 = 38
    h6 = 60
    p6 = 0

    #Variables and list for data  (d stands for day)
    dd=[]
    hdd=[]
    td=[]
    hd=[]
    pd=[]

    #Variables and list for data  (r stands for representation)
    # DR=[]
    # HRR=[]
    # TR=[]
    # HR=[]
    # PR=[]

    #Calculating Maximum and minimum temperatures
    MxT=[]
    MnT=[]

    poly = PolynomialFeatures(degree=4)
    # p=int(input("Tell whether it is raining presently or not, 0 for no and 1 for yes: "))
    p = 0
    for d in range(0,11):
        temp = str(da) + " " + str(mn) + " " + str(hr)
        for h in range(0,5):
            
            #TP,PP,HP are for 1 hour in future data
            
            dewpoint=getdata.Tdew(T,RH)
            # print("Dew Point at",hr,da,mn,"is equal to",dewpoint)
                    
            heatindex=getdata.HI(T,RH)
            # print("Heat Index at",hr,da,mn,"is equal to",heatindex)
            
            t_=np.array([da,mn,hr,heatindex,RH,P,t6]).reshape(1,-1)
            TP=list(TEMP.predict(t_))[0]
            #TP=list(TEMP.predict(poly.fit_transform(t_)))[0]
            # print("Temperature predicted at",hr,da,mn,"is equal to",TP)
            
            p_=np.array([da,mn,hr,dewpoint,RH,1,p6]).reshape(1,-1)
            PP=list(PREC.predict(p_))[0]
            # print("Precipitation predicted at",hr,da,mn,"is equal to",PP)
            
            h_=np.array([da,mn,hr,dewpoint,heatindex,T,P,h6]).reshape(1,-1)
            HP=list(HUM.predict(h_))[0]
            #HP=abs(list(HUM.predict(poly.fit_transform(h_)))[0])
            
            #print("Humidity predicted at",hr,da,mn,"is equal to",HP)
            
            #print("")
            #print("")
            
            #Adding prediction for a day list to take the mean
            dd.append(dewpoint)
            hdd.append(heatindex)
            td.append(TP)
            hd.append(HP)
            pd.append(PP)
            
            #Upddating THE previous variables
            t6=T
            h6=RH
            p6=P
                    
            T=TP
            RH=HP
            P=PP
                    
            #Updating Date and Time is final
            hr+=6
            if hr>=24:
                hr=0
                h=24
                da+=1
            if da==31:
                mn+=1
                da=1
            # key_val[temp] = {
            #     "Max. Temp": MxT[d],
            #     "Min. Temp": MnT[d],
            #     "Precipitation": PP[h],
            #     "Dew Point": DR[h],
            #     "Heat Index": HRR[h],
            #     "Relative Humidity": HP[h]
            # }    
        
        # Writing data
        
        data = (temp, TP, HP, PP)
        DATA_F += data
        #Calculating mean of all the variables for a day
        DR.append(np.mean(dd))
        HRR.append(np.mean(hdd))
        TR.append(np.mean(td))
        HR.append(np.mean(hd))
        PR.append(np.mean(pd))
        
        #Calculating Maximum and Minimum Temperature
        MxT.append(max(td))
        MnT.append(min(td))
        
        dd.clear()
        hdd.clear()
        td.clear()
        hd.clear()
        pd.clear()

    for i in range(11):
        temp = str(da) + " " + str(mn) + " " + str(hr)
        if PR[i] > 0:
            # values.append((MxT[i], MnT[i], PR[i], DR[i], HRR[i], HR[i]))
            key_val[temp] = {
                "Max. Temp": round(MxT[i], 2),
                "Min. Temp": round(MnT[i], 2),
                "Precipitation": round(PR[i], 2),
                "Dew Point": round(DR[i], 2),
                "Heat Index": round(   HRR[i], 2),
                "Relative Humidity": round(HR[i], 2)
            }
        da+=1
        if da==31:
            mn = mn+1
            da=1
    Day=np.arange(da, da+11)
    DATA_F = (MxT, MnT, PR, DR, HRR, HR)
    titles = ("Max. Temp", "Min. Temp",  "Precipitation", "Dew Point", "Heat Index", "Relative Humidity")
    graphJSON = {}
    for i in range(6):
        print(len(DATA_F[i]), i)
        fig = px.line(x=Day, y=DATA_F[i][:11], title=titles[i])
        
        graphJSON[i] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return key_val, graphJSON

def graphs():
    df2 = ps.concat({
        k: ps.DataFrame.from_dict(v, 'index') for k, v in key_val.items()
    }, axis=0)
    date = datetime.now()
    da=date.day
    Day=np.arange(da, da+11)
    print(MxT)
    
    

vals = ml()
