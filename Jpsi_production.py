import numpy as np
from iminuit import Minuit
from scipy.integrate import quad
from Evolution import Evo_WilsonCoef_SG,AlphaS
import pandas as pd
import time

NF=4

Mproton = 0.938
MJpsi = 3.097
Mcharm = MJpsi/2
alphaEM = 1/133
alphaS = 0.30187
psi2 = 1.0952 /(4 * np.pi)

conv = 2.5682 * 10 ** (-6)

def Kpsi(W :float):
    return np.sqrt(((W**2 -(MJpsi - Mproton)**2) *( W**2 -(MJpsi + Mproton)**2 ))/(4.*W**2))

def PCM(W: float):
    return np.sqrt(( W**2 - Mproton**2 )**2/(4.*W**2))

def tmin(W: float):
    return 2* Mproton ** 2 - 2 * np.sqrt((Mproton**2 + Kpsi(W)**2)*(Mproton**2 + PCM(W)**2)) - 2 * Kpsi(W) * PCM(W)

def tmax(W: float):
    return 2* Mproton ** 2 - 2 * np.sqrt((Mproton**2 + Kpsi(W)**2)*(Mproton**2 + PCM(W)**2)) + 2 * Kpsi(W) * PCM(W)

def PPlus(W: float):
    return W/np.sqrt(2)

def PprimePlus(W: float, t: float):
    return np.sqrt(Mproton**2 + Kpsi(W)**2)/np.sqrt(2) + (-2*Mproton**2 + t + 2*np.sqrt((Mproton**2 + Kpsi(W)**2)*(Mproton**2 + PCM(W)**2)))/(2.*np.sqrt(2)*PCM(W))

def PbarPlus2(W: float, t: float):
    return ( PPlus(W) + PprimePlus(W,t) ) ** 2 / 4

def DeltaPlus2(W: float, t: float):
    return (PprimePlus(W,t) - PPlus(W) ) ** 2

def Xi(W: float, t: float):
    return (PPlus(W) - PprimePlus(W,t))/(PPlus(W) + PprimePlus(W,t))


A0Lat = 0.429
MALat = 1.64
C0Lat = -1.93 /4
MCLat = 1.07

def FormFactors(t: float, A0: float, Mpole: float):
    return A0/(1 - t / (Mpole ** 2)) ** 3

def ComptonFormFactors(t: float, A0: float, MA: float, C0: float, MC: float, xi: float):

    Aformfact = FormFactors(t,A0,MA)
    Cformfact = FormFactors(t,C0,MC)
    Bformfact = 0
    
    Hformfact = Aformfact + 4 * xi**2 * Cformfact
    Eformfact = Bformfact - 4 * xi**2 * Cformfact
    
    return np.array([Hformfact, Eformfact])

def G2_New(W: float, t: float, Ag0: float, MAg: float, Cg0: float, MCg: float, Aq0: float, MAq: float, Cq0: float, MCq: float, P_order = 1): 
    xi = Xi(W ,t)
    [gHCFF, gECFF] = 2*ComptonFormFactors(t, Ag0, MAg, Cg0, MCg, xi) / xi ** 2
    [qHCFF, qECFF] = 0*2*ComptonFormFactors(t, Aq0, MAq, Cq0, MCq, xi) / xi ** 2
    
    CWS, CWG = Evo_WilsonCoef_SG(Mcharm,NF,p = 1,p_order= P_order)
    
    HCFF = CWG * gHCFF + CWS * qHCFF
    ECFF = CWG * gECFF + CWS * qECFF
    
    return (1-xi ** 2) * (HCFF + ECFF) ** 2 - 2 * ECFF * (HCFF+ECFF) + (1- t/ (4 * Mproton ** 2))* ECFF ** 2

def dsigma_New(W: float, t: float, Ag0: float, MAg: float, Cg0: float, MCg: float, Aq0: float, MAq: float, Cq0: float, MCq: float, P_order = 1):
    return 1/conv * alphaEM * (2/3) **2 /(4* (W ** 2 - Mproton ** 2) ** 2) * (16 * np.pi) ** 2/ (3 * MJpsi ** 3) * psi2 * G2_New(W, t, Ag0, MAg, Cg0, MCg, Aq0, MAq, Cq0, MCq, P_order)

def sigma_New(W: float, Ag0: float, MAg: float, Cg0: float, MCg: float, Aq0: float, MAq: float, Cq0: float, MCq: float, P_order = 1):
    return quad(lambda u: dsigma(W, u, Ag0, MAg, Cg0, MCg, Aq0, MAq, Cq0, MCq, P_order), tmin(W), tmax(W))[0]

def G2(W: float, t: float, A0: float, MA: float, C0: float, MC: float): 
    return (5/4)** 2 * 4* Xi(W ,t) ** (-4) * ((1- t/ (4 * Mproton ** 2))* FormFactors(t, C0, MC) ** 2 * (4 * Xi(W ,t) ** 2) ** 2 + 2* FormFactors(t, A0, MA) * FormFactors(t, C0, MC)*4 * Xi(W ,t) ** 2 + (1- Xi(W ,t) ** 2) * FormFactors(t,A0,MA) **2)

def dsigma(W: float, t: float, A0: float, MA: float, C0: float, MC: float):
    return 1/conv * alphaEM * (2/3) **2 /(4* (W ** 2 - Mproton ** 2) ** 2) * (16 * np.pi * alphaS)** 2/ (3 * MJpsi ** 3) * psi2 * G2(W, t, A0, MA, C0, MC)

def sigma(W: float, A0: float, MA: float, C0: float, MC: float):
    return quad(lambda u: dsigma(W, u, A0, MA, C0, MC), tmin(W), tmax(W))[0]

def WEb(Eb: float):
    return np.sqrt(Mproton)*np.sqrt(Mproton + 2 * Eb)

print(dsigma(4.58,-2,A0Lat,MALat,C0Lat,MCLat)/alphaS**2 * AlphaS(2,NF,Mcharm)**2)

print(dsigma_New(4.58,-2,A0Lat,MALat,C0Lat,MCLat, 1,1,1,1))
'''
#Read the csv into dataframe using pandas
dsigmadata = pd.read_csv("2022-final-xsec-electron-channel_total.csv")
# Not fitting the total cross-sections but I imported anyway
totsigmadata = pd.read_csv("GlueX_tot_combined.csv")

# Taking out the column that we needed
avg_E_col_dsigma=dsigmadata['avg_E'].to_numpy()
avg_abs_t_col_dsigma = dsigmadata['avg_abs_t'].to_numpy()
dsdt_nb_col_dsigma = dsigmadata['dsdt_nb'].to_numpy()
tot_error_col_dsigma = dsigmadata['tot_error'].to_numpy()

# Calculate the W in terms of the beam energy for the whole array
avg_W_col_dsigma = WEb(avg_E_col_dsigma)

# Creat a 2d array shape (N,4) with each row (W,|t|,dsigma,dsigma_err)
dsigmadata_reshape = np.column_stack((avg_W_col_dsigma, avg_abs_t_col_dsigma, dsdt_nb_col_dsigma, tot_error_col_dsigma))

# We want to select all the data with xi > xi_thres, here I put xi_thres = 0.5 to be consist with the paper
xi_thres = 0.5
# calculate the xi for each row/data point
xi_col_dsigma = Xi(avg_W_col_dsigma, -avg_abs_t_col_dsigma)
# Creat a mask that the condition is met
mask = xi_col_dsigma>=xi_thres 
# Select the data with the mas
dsigmadata_select = dsigmadata_reshape[mask]
# only 33 data left with xi>0.5
print(dsigmadata_select.shape[0])

#
# The same thing for the total cross-sections (Not fitted)
#
# Taking out the column that we needed
avg_E_col_sigma = totsigmadata['E_avg'].to_numpy()
sigma_col_sigma = totsigmadata['sigma'].to_numpy()
sigma_err_col_sigma = totsigmadata['sigma_err'].to_numpy()

# Calculate the W in terms of the beam energy for the whole array
avg_W_col_sigma = WEb(avg_E_col_sigma)

# Creat a 2d array shape (N,3) with each row (W,dsigma,dsigma_err)
totsigmadata_reshape =  np.column_stack((avg_W_col_sigma,sigma_col_sigma,sigma_err_col_sigma))

def chi2(A0: float, MA: float, C0: float, MC: float):

    #sigma_pred = list(map(lambda W: sigma(W, A0, MA, C0, MC), totsigmadata_reshape[:,0]))
    #chi2sigma = np.sum(((sigma_pred - totsigmadata_reshape[:,1]) / totsigmadata_reshape[:,2]) **2 )

    #Two variables Wt[0] = W, Wt[1] = |t| = -t
    dsigma_pred=list(map(lambda Wt: dsigma(Wt[0], -Wt[1], A0, MA, C0, MC), zip(dsigmadata_select[:,0], dsigmadata_select[:,1])))
    chi2dsigma = np.sum(((dsigma_pred - dsigmadata_select[:,2]) / dsigmadata_select[:,3]) **2 )
    return chi2dsigma #+ chi2sigma
'''

'''
time_start = time.time()

A0pdf = 0.414
m = Minuit(chi2, A0 = A0pdf, MA = MALat, C0 = C0Lat ,MC = MCLat)
m.errordef = 1
#m.fixed["A0"] = True
#m.fixed["MC"] = True
m.fixed["A0"] = True
m.limits["C0"] = (-20,20)
m.migrad()
m.hesse()

ndof = dsigmadata_select.shape[0]  - m.nfit  # + totsigmadata_reshape.shape[0]

time_end = time.time() -time_start

with open('FitOutput.txt', 'w', encoding='utf-8', newline='') as f:
    print('Total running time: %.1f minutes. Total call of cost function: %3d.\n' % ( time_end/60, m.nfcn), file=f)
    print('The chi squared/d.o.f. is: %.2f / %3d ( = %.2f ).\n' % (m.fval, ndof, m.fval/ndof), file = f)
    print('Below are the final output parameters from iMinuit:', file = f)
    print(*m.values, sep=", ", file = f)
    print(*m.errors, sep=", ", file = f)
    print(m.params, file = f)
'''