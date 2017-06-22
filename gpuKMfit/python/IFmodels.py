import numpy as np
from scipy.optimize import least_squares
import plot_functions as pf

# Evaluate Feng input function (FDG)
# y = FengInput(parm, t)
# parm =  [modelNumber delay a(1:end) lambda(1:end)];
#      or [modelNumber delay];
#      or [modelNumber];
#      or [];
# in the latter cases, default values will be used
# t = time in minutes
# y = input function
#
# Example usage:
# t=0:60;
# plot(t,FengInput(2,t))
#
# D Feng, SC Huang, X Wang
# Models for Computer simulation Studies of Input Functions for 
# Tracer Kinetic Modeling with Positron emission Tomography
# International Journal of Biomedical Computing 32(2):95-110, 1993 (March).
# (Int J of Biomed Comput)

def simulate(scanTime, IFparams, model=2):
    time = set_time(scanTime)
    IFparams = list(IFparams)
    IFparams.insert(0,model)
    return fengInput(np.asarray(IFparams), time)
    
def fit(IF_meas, scanTime, model=2):    
    time = set_time(scanTime)
    IF = np.asarray(IF_meas)
    order = np.double(np.rint(np.log(IF.max())/np.log(10))) + 1
    print IF.max(), order
    #if order<=0: 
    #   order=-0.5
    
    if model==2 or model == 3:
      x0=[0, 10*(10**order), 0.05*(10**order), 0.1*(10**order), -5, -0.1, -0.01]
      lb = [0,0,0,0,-np.Inf,-np.Inf,-np.Inf]
      ub = [1,np.Inf,np.Inf,np.Inf,0,0,0]
    elif model==4 or model==6:
      x0=[0, 10*(10**order), 0.05*(10**order), -5., -0.1]
      lb = [0,0,0,-np.Inf,-np.Inf]
      ub = [1,np.Inf,np.Inf,0,0]
    elif model == 5:
      x0=[0, 10*(10**order), -5, -0.1]
      lb = [0,0,-np.Inf,-np.Inf]
      ub = [1,np.Inf,0,0]
    
    fitting = least_squares(residual, x0, bounds=(lb, ub), max_nfev=1e16, loss='linear', f_scale=1.,args=(time, model, IF))    
    return fitting.x

    
######################### SIMULATE MODELS ############################
 
def set_time(scanTime): 
    scanTime = np.asarray(scanTime)
    if scanTime.max()>180:
        scanTime = scanTime/60; #time has to be in minutes
    return np.mean(scanTime,0)
    
def fengInput(parm, t):
    
    if parm.size < 1:
        modelNumber = 2
    else:
        modelNumber = int(parm[0])

    if modelNumber == 2:
        if parm.size < 2:
            delay = 0.735
        else:
            delay = parm[1]
        if parm.size < 3:
            a = [851.1, 21.88, 20.81]
            l = [-4.134, -0.1191, -0.01043]
        else:
            a=parm[2:5]
            l=parm[5:8]

        y = np.zeros(t.size)            
        idx = np.where(t>=delay)[0]
             
        y[idx] = (a[0]*(t[idx]-delay)-a[1]-a[2])*np.exp(l[0]*(t[idx]-delay)) + a[1]*np.exp(l[1]*(t[idx]-delay)) + a[2]*np.exp(l[2]*(t[idx]-delay))


    elif modelNumber == 3:
            delay = 0;
            if parm.size < 3:
                a = [851.1, 21.88, 20.81]
                l = [-4.134, -0.1191, -0.01043]
            else:
                a=parm[2:5]
                l=parm[5:8]

            y = np.zeros(t.size)            
            idx = np.where(t>=delay)[0]
            
            y[idx] = (a[0]*(t[idx]-delay)-a[1]-a[2])*np.exp(l[0]*(t[idx]-delay)) + a[1]*np.exp(l[1]*(t[idx]-delay)) + a[2]*np.exp(l[2]*(t[idx]-delay))


    elif modelNumber == 4:
            if parm.size < 2:
                delay = 0.731
            else:
                delay = parm[1]

            if parm.size < 3:
                a = [892.5, 36.8]
                l = [-3.8862, -0.0262]
            else:
                a = parm[2:4]
                l = parm[4:6]

            y = np.zeros(t.size)            
            idx = np.where(t>=delay)[0]
            y[idx] =  (a[0]*(t[idx]-delay)-a[1])*np.exp(l[0]*(t[idx]-delay)) + a[1]*np.exp(l[1]*(t[idx]-delay))

    elif modelNumber == 5:
            if parm.size < 2:
                delay = 0.781
            else:
                delay = parm[1]

            if length(parm) < 3:
                a = [90.64]
                l = [-77.53, -0.341]
            else:
                a = parm[2]
                l = parm[3:5]

            y = np.zeros(t.size)            
            idx = np.where(t>=delay)[0]
            y[idx] = a[0]*np.exp(l[0]*(t[idx]-delay)) + a[0]*np.exp(l[1]*(t[idx]-delay))      # note the second term is + here
                                                                                        # and - in eq. 5 of Feng 1993
                                                                                        # this must be a typo in Feng 1993

    else:
            print 'FengInput: model '  + str(modelNumber) +' not implemented'
            y = []

    return y



######################### FIT MODELS ############################

def residual(par, time, model, measurement):
    p = list(par)
    p.insert(0,model)
    fit = fengInput(np.asarray(p), time)
    return np.asarray(fit - measurement)

######################### TEST MODELS ############################
def test_example():
    time = [[0.,0.16666667,0.33333333,0.5,0.66666667,0.83333333,1.,1.16666667,1.33333333,
             1.5,1.66666667,1.83333333,2.,2.5,3.,4.,5.,6.,8.,10.,15.,20.,25.,30.],
            [0.16666667,0.33333333,0.5,0.66666667,0.83333333,1.,1.16666667,1.33333333,1.5,
             1.66666667,1.83333333,2.,2.5,3.,4.,5.,6.,8.,10.,15.,20.,25.,30.,40.,]]
    

    IF = simulate(time, [0.3, 851.1*300, 21.88*100, 20.81*500, -4.134, -0.1191, -0.01043], model=2)
    pf.plot_TAC_and_IF(set_time(time), IF, IF)

#########################################################################################################
def test_simulation(): 
    
    time = [[0.,0.16666667,0.33333333,0.5,0.66666667,0.83333333,1.,1.16666667,1.33333333,
             1.5,1.66666667,1.83333333,2.,2.5,3.,4.,5.,6.,8.,10.,15.,20.,25.,30.],
            [0.16666667,0.33333333,0.5,0.66666667,0.83333333,1.,1.16666667,1.33333333,1.5,
             1.66666667,1.83333333,2.,2.5,3.,4.,5.,6.,8.,10.,15.,20.,25.,30.,40.,]]
    
    IFparams = [0.458300896195880, 
                758028.906510941, 3356.00773871079, 7042.64861309165, 
                -9.91821801288336, 0.0134846319687693, -0.0585800774301212]
    
    Cp = [0.,0.,24409.38070751,29004.28711479,  16902.29913917,12060.98208071,10612.57271934,10197.68263123,
          10054.6062693,9978.15052684,9917.65977008,9861.25459037,9698.43991838,9541.40325626,9243.27864281,
          8965.43931906,8706.77619679,8242.85724982,7843.85989999,7087.59694672,6609.51306567,6344.98718338,
          6246.22490223,6414.56761034]
    
    IF = simulate(time, IFparams, model=2)
    pf.plot_TAC_and_IF(set_time(time), Cp, IF)
    

#########################################################################################################
def test_optimizer():
    
    time = [[0.,0.16666667,0.33333333,0.5,0.66666667,0.83333333,1.,1.16666667,1.33333333,
             1.5,1.66666667,1.83333333,2.,2.5,3.,4.,5.,6.,8.,10.,15.,20.,25.,30.],
            [0.16666667,0.33333333,0.5,0.66666667,0.83333333,1.,1.16666667,1.33333333,1.5,
             1.66666667,1.83333333,2.,2.5,3.,4.,5.,6.,8.,10.,15.,20.,25.,30.,40.]]
    
    IF0 = [0,2478.83333333333,24717.7500000000,23788.7500000000,9176.75000000000,7598.33333333333,5912.33333333333,
           7053.58333333333,6036.41666666667,6931.66666666667,6430.66666666667,4734.16666666667,5992.75000000000,
           5328.33333333333,5910,4780.58333333333,4957.08333333333,4907.66666666667,5176.83333333333,3978.66666666667,
           3975.75000000000,3720,3683.25000000000,3640.25000000000]
    
    IF2 = [0,0,2505.19877675841,19728.4403669725,7828.74617737003,5480.12232415902,5323.54740061162,4697.24770642202,
           3601.22324159021,2661.77370030581,3444.64831804281,4227.52293577982,3601.22324159021,4018.75637104995,
           3261.97757390418,3261.97757390418,4331.90621814475,4110.09174311927,3653.41488277268,2834.00611620795,
           2552.17125382263,2411.25382262997,2359.06218144750,2022.42609582059]
    
    IF3 = [0,0,486.666666666667,27818.5416666667,22800.2916666667,11035.4166666667,12024.3333333333,8327.95833333333,
           5654.95833333333,7515.29166666667,8963.29166666667,8924.83333333333,10006.1666666667,7740.83333333333,
           9368.04166666667,8876.95833333333,7358.29166666667,7238.91666666667,7833.20833333333,7107.87500000000,
           6946.16666666667,6792.12500000000,6805.20833333333,5709.25000000000]
    
    IF4 = [0,0,2549.51428571429,29643.4285714286,29409.3428571429,15735.7142857143,11130.8285714286,12903.2000000000,
           13519.3428571429,10451.8857142857,9091.57142857143,10514.8571428571,10036.7428571429,9627.28571428571,
           10660.4857142857,9705,9068.40000000000,7706.65714285714,8759.65714285714,7410.25714285714,6975.37142857143,
           6309.77142857143,5927.51428571429,5750.14285714286]
    
    IF5 = [0,0,11867.5666666667,22517.0666666667,13853.8000000000,8516.03333333333,9065.53333333333,10561.9666666667,
           9383.80000000000,7401,7617.33333333333,7791.50000000000,5941.46666666667,7812.93333333333,7210.36666666667,
           8728.16666666667,7956.50000000000,7905.20000000000,7441.63333333333,7066.73333333333,6886.80000000000,6809.26666666667,
           7050.80000000000,6202.33333333333]
    
    DATA = np.asarray([IF0,IF2,IF3,IF4,IF5])
    #DATA = np.asarray([IF5])
    
    for curve in DATA:
        params = fit(curve, time, model=2)
        IFfit = simulate(time, params, 2)
        pf.plot_TAC_and_IF(set_time(time), IFfit, curve)


    
    
#########################################################################################################   
if __name__ == "__main__": 
    test_simulation() 
    
    
    
    
    
    
    