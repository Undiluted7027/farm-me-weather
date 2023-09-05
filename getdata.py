# Importing required modules
import math, time, serial

def serial_req(port, baud_rate):
    ser = serial.Serial(port, baud_rate) # connecting with arduino port at baud_rate
    time.sleep(2) # To avoid serial buffer connection complications
    flag = True
    while True:
        bit = ser.readline().decode() # read incoming byte string and decode it
        string = bit.rstrip() # purify string
        if len(string.split(" ")) == 8:
            print(string)
            break
        time.sleep(0.1) # to make sure buffer overflow isn't there
    ser.close()
    data = string.split(" ")
    data = tuple([float(i) for i in data])
    return data

# Dew Point
def ln(n):
    return math.log(n,math.e)

def Tdew(T,H):
    nl = ln(H/100)
    s = 237.3+T
    m = 17.27*T
    TD = (237.3*(nl+(m/s)))/(17.27+(-1)*(nl+(m/s)))    
    return TD

# Heat Index
def HI(t,H):
    c1 = -42.379
    c2 = 2.04901523
    c3 = 10.14333127
    c4 = -0.22475541
    c5 = -6.83783 *(0.001)
    c6 = -5.481717 * (0.01)
    c7 = 1.22874 * (0.001)
    c8 = 8.5282 * (0.0001)
    c9 = -1.99 * (0.000001)
    T=(1.8*t)+32
    HI=c1+(c2*T)+(c3*H)+(c4*T*H)+(c5*T*T)+(c6*H*H)+(c7*T*H*T)+(c8*T*H*H)+(c9*T*H*T*H)
    FHI=((HI-32)*5)/9
    return FHI



