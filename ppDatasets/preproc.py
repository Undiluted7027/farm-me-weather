import math
import pandas as pd

LISTS = ["preci", "humi", "temp"]

def HI(t,H):
    c1 = -42.379
    c2= 2.04901523
    c3 = 10.14333127
    c4 = -0.22475541
    c5 = -6.83783 *(0.001)
    c6 = -5.481717 * (0.01)
    c7 = 1.22874 * (0.001)
    c8 = 8.5282 * (0.0001)
    c9 = -1.99 * (0.000001)
    T=(1.8*t)+32
    H_I=c1+(c2*T)+(c3*H)+(c4*T*H)+(c5*T*T)+(c6*H*H)+(c7*T*H*T)+(c8*T*H*H)+(c9*T*H*T*H)
    FHI=((H_I-32)*5)/9
    return FHI

def replace(index, df, val, col):
    if df[col][index]!=(val): # -10
        return df[col][index]
    else:
        return replace(index+1, df, val, col)

def ln(n):
    return math.log(n,math.e)

def Tdew(T,H):
    nl =ln(H/100)
    s=237.3+T
    m=17.27*T
    TD = (237.3*(nl+(m/s)))/(17.27+(-1)*(nl+(m/s)))    
    return TD

def ppHumi(df):
    #df = set_up()
    df["temp"].fillna(-10, inplace = True)

    for i in range(100990):
        if df["temp"][i]==-10:
            df["temp"][i]=(replace(i+1, df, -10, "temp")+df["temp"][i-1])/2
        elif df["temp"][i]>=50:
            df["temp"][i]=(df["temp"][i-1]+df["temp"][i+1])/2

    df["humidity previous"].fillna(-100, inplace = True)

    for i in range(100990):
        if df["humidity previous"][i]==-100:
            df["humidity previous"][i]=(replace(i+1, df, -100, "humidity previous")+df["humidity previous"][i-1])/2
        elif df["humidity previous"][i]>100:
            df["humidity previous"][i]=(df["humidity previous"][i-1]+df["humidity previous"][i+1])/2

    df["humidity"].fillna(-300, inplace = True)

    for i in range(100990):
        if df["humidity"][i]==-300:
            df["humidity"][i]=df["humidity previous"][i+1]
        elif df["humidity"][i]>100:
            df["humidity"][i]=df["humidity previous"][i+1]
    
    df["heatindex"].fillna(-200, inplace = True)

    for i in range(100990):
        if df["heatindex"][i]==-200:
            df["heatindex"][i]=HI(df["temp"][i],df["humidity previous"][i])
        elif df["heatindex"][i]>60:
            df["heatindex"][i]=HI(df["temp"][i],df["humidity previous"][i])

    df["dewpt"].fillna(-500, inplace = True)
    for i in range(100990):
        if df["dewpt"][i]==-500:
            df["dewpt"][i]=Tdew(df["temp"][i],df["humidity previous"][i])
        elif df["dewpt"][i]>40 or df["dewpt"][i]<(-12):
            df["dewpt"][i]=Tdew(df["temp"][i],df["humidity previous"][i])
    
    humidity_num=df[list(df.dtypes[df.dtypes!='object'].index)]
    humidity_y=humidity_num.pop('humidity')
    humidity_x=humidity_num

    return df, humidity_x, humidity_y


def ppTemp(df):
    df.drop(["Time"],axis=1,inplace=True)
    df["temp previous"].fillna(-10, inplace = True)

    for i in range(100990):
        if df["temp previous"][i]==-10:
            df["temp previous"][i]=(replace(i+1, df, -10, "temp previous")+df["temp previous"][i-1])/2
        elif df["temp previous"][i]>=50:
            df["temp previous"][i]=(df["temp previous"][i-1]+df["temp previous"][i+1])/2

    df["temp"].fillna(-300, inplace = True)

    for i in range(100990):
        if df["temp"][i]==-300:
            df["temp"][i]=df["temp previous"][i+1]
        elif df["temp"][i]>100:
            df["temp"][i]=df["temp previous"][i+1]

    df["humidity"].fillna(-100, inplace = True)

    for i in range(100990):
        if df["humidity"][i]==-100:
            df["humidity"][i]=(replace(i+1, df, -100, "humidity")+df["humidity"][i-1])/2
        elif df["humidity"][i]>100:
            df["humidity"][i]=(df["humidity"][i-1]+df["humidity"][i+1])/2

    df["heatindex"].fillna(-200, inplace = True)

    for i in range(100990):
        if df["heatindex"][i]==-200:
            df["heatindex"][i]=HI(df["temp previous"][i],df["humidity"][i])
        elif df["heatindex"][i]>60:
            df["heatindex"][i]=HI(df["temp previous"][i],df["humidity"][i])

    weather_df_num=df[list(df.dtypes[df.dtypes!='object'].index)]
    df_y = weather_df_num.pop('temp')
    df_x = weather_df_num
    print(df.isnull().any())
    return df, df_x, df_y


def ppPreci(df):
    df["temp"].fillna(-10, inplace = True)

    for i in range(100990):
        if df["temp"][i]==-10:
            df["temp"][i]=(replace(i+1, df, -10, "temp")+df["temp"][i-1])/2
        elif df["temp"][i]>=50:
            df["temp"][i]=(df["temp"][i-1]+df["temp"][i+1])/2
    
    df["humidity"].fillna(-100, inplace = True)

    for i in range(100990):
        if df["humidity"][i]==-100:
            df["humidity"][i]=(replace(i+1, df, -100, "humidity")+df["humidity"][i-1])/2
        elif df["humidity"][i]>100:
            df["humidity"][i]=(df["humidity"][i-1]+df["humidity"][i+1])/2
    
    df.drop(["Time"],axis=1,inplace=True)
    df["dewpt"].fillna(-500, inplace = True)

    for i in range(100990):
        if df["dewpt"][i]==-500:
            df["dewpt"][i]=Tdew(df["temp"][i],df["humidity"][i])
        elif df["dewpt"][i]>40 or df["dewpt"][i]<(-12):
            df["dewpt"][i]=Tdew(df["temp"][i],df["humidity"][i])

    df.drop(["temp"],axis=1,inplace=True)
    df["Date"].fillna(-500, inplace = True)
    df["Hour"].fillna(-500, inplace = True)
    df["Month"].fillna(-500, inplace = True)
    df["rain"].fillna(-500, inplace = True)
    df["thunder"].fillna(-500, inplace = True)
    df["rain previous"].fillna(-500, inplace = True)
    df_x = df.drop(["rain"],axis=1)
    df_y = df["rain"]

    return df, df_x, df_y

def set_up(key):
    if key == "temp":
        df1 = pd.read_csv(r"Final Datasets/nTemperature.csv")
        df1, df1_x, df1_y = ppTemp(df1)
        df1_x.to_csv("ppTempX.csv", index=False)
        df1_y.to_csv("ppTempY.csv", index=False)
        df1.to_csv("ppTemp.csv", index=False)
        print(f"{key} dataset has been preprocessed successfully!")
        return df1_x, df1_y
        
    elif key == "humi":
        df2 = pd.read_csv(r"Final Datasets/nHumidity.csv")
        df2_x, df2_y, df2 = ppHumi(df2)
        df2_x.to_csv("ppHumX.csv", index=False)
        df2_y.to_csv("ppHumY.csv", index=False)
        df2.to_csv("ppHum.csv", index=False)
        print(f"{key} dataset has been preprocessed successfully!")
        return df2_x, df2_y

    elif key == "preci":
        df3 = pd.read_csv(r"Final Datasets/nRainfall.csv")
        df3_x, df3_y, df3 = ppPreci(df3)
        df3_x.to_csv("ppRainX.csv", index=False)
        df3_y.to_csv("ppRainY.csv", index=False)
        df3.to_csv("ppRain.csv", index=False)
        print(f"{key} dataset has been preprocessed successfully!")
        return df3_x, df3_y

set_up("temp")
set_up("humi")
set_up("preci")