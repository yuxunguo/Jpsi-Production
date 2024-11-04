import numpy as np
from iminuit import Minuit
from scipy.integrate import quad
from Evolution import Evo_WilsonCoef_SG,AlphaS
import pandas as pd
import time
import os

import matplotlib.pyplot as plt

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

def WEb(Eb: float):
    return np.sqrt(Mproton)*np.sqrt(Mproton + 2 * Eb)

def FormFactors(t: float, A0: float, Mpole: float, p_pole: int):
    return A0/(1 - t / (Mpole ** 2)) ** p_pole

def ComptonFormFactors(t: float, A0: float, MA: float, C0: float, MC: float, xi: float, A_pole: int, C_pole: int):

    Aformfact = FormFactors(t,A0,MA,A_pole)
    Cformfact = FormFactors(t,C0,MC,C_pole)
    Bformfact = 0
    
    Hformfact = Aformfact + 4 * xi**2 * Cformfact
    Eformfact = Bformfact - 4 * xi**2 * Cformfact
    
    return np.array([Hformfact, Eformfact])

def G2_New(W: float, t: float, Ag0: float, MAg: float, Cg0: float, MCg: float, Aq0: float, MAq: float, Cq0: float, MCq: float, P_order: int = 1, A_pole: int = 2, C_pole: int = 3): 
    xi = Xi(W ,t)
    [gHCFF, gECFF] = 2*ComptonFormFactors(t, Ag0, MAg, Cg0, MCg, xi, A_pole, C_pole) / xi ** 2
    [qHCFF, qECFF] = 2*ComptonFormFactors(t, Aq0, MAq, Cq0, MCq, xi, A_pole, C_pole) / xi ** 2
    
    CWS, CWG = np.real(Evo_WilsonCoef_SG(Mcharm,NF,p = 1,p_order= P_order))
    
    HCFF = CWG * gHCFF + CWS * qHCFF
    ECFF = CWG * gECFF + CWS * qECFF
    
    return (1-xi ** 2) * (HCFF + ECFF) ** 2 - 2 * ECFF * (HCFF+ECFF) + (1- t/ (4 * Mproton ** 2))* ECFF ** 2

def dsigma_New(W: float, t: float, Ag0: float, MAg: float, Cg0: float, MCg: float, Aq0: float, MAq: float, Cq0: float, MCq: float, P_order = 1, A_pole: int = 2, C_pole: int = 3):
    return 1/conv * alphaEM * (2/3) **2 /(4* (W ** 2 - Mproton ** 2) ** 2) * (16 * np.pi) ** 2/ (3 * MJpsi ** 3) * psi2 * G2_New(W, t, Ag0, MAg, Cg0, MCg, Aq0, MAq, Cq0, MCq, P_order, A_pole, C_pole)

def sigma_New(W: float, Ag0: float, MAg: float, Cg0: float, MCg: float, Aq0: float, MAq: float, Cq0: float, MCq: float, P_order = 1, A_pole: int = 2, C_pole: int = 3):
    return quad(lambda u: dsigma(W, u, Ag0, MAg, Cg0, MCg, Aq0, MAq, Cq0, MCq, P_order, A_pole, C_pole), tmin(W), tmax(W))[0]

def G2(W: float, t: float, A0: float, MA: float, C0: float, MC: float): 
    return (5/4)** 2 * 4* Xi(W ,t) ** (-4) * ((1- t/ (4 * Mproton ** 2))* FormFactors(t, C0, MC) ** 2 * (4 * Xi(W ,t) ** 2) ** 2 + 2* FormFactors(t, A0, MA) * FormFactors(t, C0, MC)*4 * Xi(W ,t) ** 2 + (1- Xi(W ,t) ** 2) * FormFactors(t,A0,MA) **2)

def dsigma(W: float, t: float, A0: float, MA: float, C0: float, MC: float):
    return 1/conv * alphaEM * (2/3) **2 /(4* (W ** 2 - Mproton ** 2) ** 2) * (16 * np.pi * alphaS)** 2/ (3 * MJpsi ** 3) * psi2 * G2(W, t, A0, MA, C0, MC)

def sigma(W: float, A0: float, MA: float, C0: float, MC: float):
    return quad(lambda u: dsigma(W, u, A0, MA, C0, MC), tmin(W), tmax(W))[0]



minus_t = np.array(np.load('Lattice Data/minus_t.npy'))
minus_t_D = minus_t[1:]

AgDg_mean = np.load('Lattice Data/AgDg_mean.npy')
Ag_mean = AgDg_mean[:34]
Ag_data = np.column_stack((minus_t,Ag_mean))

Dg_mean = AgDg_mean[34:]
Dg_data = np.column_stack((minus_t_D,Dg_mean))

AqDq_mean = np.load('Lattice Data/AqDq_mean.npy')
Aq_mean = AqDq_mean[:34]
Aq_data = np.column_stack((minus_t,Aq_mean))
Dq_mean = AqDq_mean[34:]
Dq_data = np.column_stack((minus_t_D,Dq_mean))

AgDg_cov = np.load('Lattice Data/AgDg_cov.npy')
AgDg_diag = np.diagonal(AgDg_cov)
Ag_err = np.sqrt(AgDg_diag[:34])
Dg_err = np.sqrt(AgDg_diag[34:])

AqDq_cov = np.load('Lattice Data/AqDq_cov.npy')
AqDq_diag = np.diagonal(AqDq_cov)
Aq_err = np.sqrt(AqDq_diag[:34])
Dq_err = np.sqrt(AqDq_diag[34:])

#print(dsigma(WEb(8.78),-3,Ag0lat,MAglat,Cg0lat,MCglat)/alphaS**2 * AlphaS(2,NF,Mcharm)**2)
Ag0lat = 0.4776
MAglat = 1.6746
Cg0lat = -0.1171
MCglat = 3.0829

Aq0lat = 0.5
MAqlat = 2.0179
Cq0lat = -0.2245
MCqlat = 1.9515

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
# Total cross-sections (Not fitted)
#
# Taking out the column that we needed
avg_E_col_sigma = totsigmadata['E_avg'].to_numpy()
sigma_col_sigma = totsigmadata['sigma'].to_numpy()
sigma_err_col_sigma = totsigmadata['sigma_err'].to_numpy()

# Calculate the W in terms of the beam energy for the whole array
avg_W_col_sigma = WEb(avg_E_col_sigma)

# Creat a 2d array shape (N,3) with each row (W,dsigma,dsigma_err)
totsigmadata_reshape =  np.column_stack((avg_W_col_sigma,sigma_col_sigma,sigma_err_col_sigma))

INCLUDE_XSEC = False
P_ORDER = 1

def chi2(Ag0: float, MAg: float, Cg0: float, MCg: float, Aq0: float, MAq: float, Cq0: float, MCq: float, A_pole: int, C_pole: int):

    #sigma_pred = list(map(lambda W: sigma(W, A0, MA, C0, MC), totsigmadata_reshape[:,0]))
    #chi2sigma = np.sum(((sigma_pred - totsigmadata_reshape[:,1]) / totsigmadata_reshape[:,2]) **2 )

    Aqlst = FormFactors(-minus_t, Aq0, MAq, A_pole) # = Aq(t0)
    Dqlst = 4 * FormFactors(-minus_t_D, Cq0, MCq, C_pole)
    
    Aglst = FormFactors(-minus_t, Ag0, MAg, A_pole) # = Aq(t0)
    Dglst = 4 * FormFactors(-minus_t_D, Cg0, MCg, C_pole)

    chi2Aq = np.sum( ((Aqlst-Aq_mean)/Aq_err) ** 2 )
    chi2Dq = np.sum( ((Dqlst-Dq_mean)/Dq_err) ** 2 )
    chi2Ag = np.sum( ((Aglst-Ag_mean)/Ag_err) ** 2 )
    chi2Dg = np.sum( ((Dglst-Dg_mean)/Dg_err) ** 2 )

    # Two variables Wt[0] = W, Wt[1] = |t| = -t
    if(INCLUDE_XSEC):
        dsigma_pred=list(map(lambda Wt: dsigma_New(Wt[0], -Wt[1], Ag0, MAg, Cg0, MCg, Aq0, MAq, Cq0, MCq, P_order = P_ORDER), zip(dsigmadata_select[:,0], dsigmadata_select[:,1])))
        chi2dsigma = np.sum(((dsigma_pred - dsigmadata_select[:,2]) / dsigmadata_select[:,3]) **2 )
    else:
        chi2dsigma = 0
    #return chi2Aq + chi2Dq + chi2Ag + chi2Dg # Fitting only to lattice 
    return chi2Aq + chi2Dq + chi2Ag + chi2Dg + chi2dsigma # + chi2sigma

def fit(str):
    time_start = time.time()

    m = Minuit(chi2, Ag0 = Ag0lat, MAg = MAglat, Cg0 = Cg0lat ,MCg = MCglat, Aq0 = Aq0lat, MAq = MAqlat, Cq0 = Cq0lat ,MCq = MCqlat, A_pole = 2, C_pole = 3)
    m.errordef = 1
    m.fixed["A_pole"] = True
    m.fixed["C_pole"] = True

    m.limits["Ag0"] = (0,1)
    m.limits["MAg"] = (0,5)
    m.limits["Cg0"] = (-5,5)
    m.limits["MCg"] = (0,5)
    m.limits["Aq0"] = (0,1)
    m.limits["MAq"] = (0,5)
    m.limits["Cq0"] = (-5,5)
    m.limits["MCq"] = (0,5)

    m.migrad()
    m.hesse()

    # ndof = Aq_mean.shape[0] + Dq_mean.shape[0] + Ag_mean.shape[0] + Dg_mean.shape[0]  - m.nfit
    ndof = Aq_mean.shape[0] + Dq_mean.shape[0] + Ag_mean.shape[0] + Dg_mean.shape[0] + dsigmadata_select.shape[0] * INCLUDE_XSEC - m.nfit  #  + totsigmadata_reshape.shape[0]

    time_end = time.time() -time_start
    
    os.makedirs(f'Output/{str}', exist_ok=True)
    
    with open(f'Output/{str}/Summary.txt', 'w', encoding='utf-8', newline='') as f:
        print('Total running time: %.1f minutes. Total call of cost function: %3d.\n' % ( time_end/60, m.nfcn), file=f)
        print('The chi squared/d.o.f. is: %.2f / %3d ( = %.2f ).\n' % (m.fval, ndof, m.fval/ndof), file = f)
        print('Below are the final output parameters from iMinuit:', file = f)
        print(*m.values, sep=", ", file = f)
        print(*m.errors, sep=", ", file = f)
        print(m.params, file = f)
        
    param_names = m.parameters[:-2]
    param_len = len(param_names)

    fig = plt.figure(figsize=(24,20))
    gs = fig.add_gridspec(param_len, param_len)

    for idx in range(param_len):
        for idy in range(idx+1,param_len):
            ax = fig.add_subplot(gs[idy, idx]) 
            #x, y, Z = m.contour(param_names[idx], param_names[idy], size=100, bound = (m.limits[param_names[idx]],m.limits[param_names[idy]]))
            x, y, Z = m.contour(param_names[idx], param_names[idy], size=50)
            ax.contour(x, y, Z,levels = 25, cmap='viridis')
            if(idy==param_len-1):
                ax.set_xlabel(param_names[idx])
            if(idx==0):
                ax.set_ylabel(param_names[idy])

    for idxx in range(param_len):
        ax = fig.add_subplot(gs[idxx, idxx]) 
        x, y = m.profile(param_names[idxx], size = 50, bound = m.limits[param_names[idxx]])
        ax.plot(x,y)
        if(idxx==param_len-1):
            ax.set_xlabel(param_names[idxx])
        if(idxx==0):
            ax.set_ylabel(param_names[idxx])
        
    plt.tight_layout()
    plt.savefig(f'Output/{str}/Correlation.png') 
    plt.close('all')
    
    Ag0_bf = m.values["Ag0"]
    MAg_bf = m.values["MAg"]
    Cg0_bf = m.values["Cg0"]
    MCg_bf = m.values["MCg"]
    
    Aq0_bf = m.values["Aq0"]
    MAq_bf = m.values["MAq"]
    Cq0_bf = m.values["Cq0"]
    MCq_bf = m.values["MCq"]
    
    A_pole_bf = m.values["A_pole"]
    C_pole_bf = m.values["C_pole"]
    
    fit_minus_t = np.linspace(0, 2, 100)
    
    Aq_bf = FormFactors(-fit_minus_t, Aq0_bf, MAq_bf, A_pole_bf)
    Ag_bf = FormFactors(-fit_minus_t, Ag0_bf, MAg_bf, A_pole_bf)
    Cq_bf = 4*FormFactors(-fit_minus_t, Cq0_bf, MCq_bf, C_pole_bf)
    Cg_bf = 4*FormFactors(-fit_minus_t, Cg0_bf, MCg_bf, C_pole_bf)
    
    fig_2 = plt.figure(figsize=(24,20))
    gs_2 = fig_2.add_gridspec(2, 2)
    
    ax11 = fig_2.add_subplot(gs_2[0, 0])
    ax11.errorbar(minus_t, Aq_mean, yerr = Aq_err, fmt='o', capsize=5, capthick=1, ecolor='red', label="Lattice $A_q$")
    ax11.plot(fit_minus_t, Aq_bf, color = 'blue', label= f'Best-fit $A_q$')
    ax11.set_xlabel('-t (GeV$^2$)')
    ax11.legend(fontsize=30)
    
    ax12 = fig_2.add_subplot(gs_2[0, 1])
    ax12.errorbar(minus_t_D, Dq_mean, yerr = Dq_err, fmt='o', capsize=5, capthick=1, ecolor='red', label="Lattice $D_q$")
    ax12.plot(fit_minus_t, Cq_bf, color = 'blue', label= f'Best-fit $D_q$')
    ax12.set_xlabel('-t (GeV$^2$)')
    ax12.legend(fontsize=30)
    
    ax21 = fig_2.add_subplot(gs_2[1, 0])
    ax21.errorbar(minus_t, Ag_mean, yerr = Ag_err, fmt='o', capsize=5, capthick=1, ecolor='red', label="Lattice $A_g$")
    ax21.plot(fit_minus_t, Ag_bf, color = 'blue', label= f'Best-fit $A_g$')
    ax21.set_xlabel('-t (GeV$^2$)')
    ax21.legend(fontsize=30)
    
    ax22 = fig_2.add_subplot(gs_2[1, 1])
    ax22.errorbar(minus_t_D, Dg_mean, yerr = Dg_err, fmt='o', capsize=5, capthick=1, ecolor='red', label="Lattice $D_g$")
    ax22.plot(fit_minus_t, Cg_bf, color = 'blue', label= f'Best-fit $D_g$')
    ax22.set_xlabel('-t (GeV$^2$)')
    ax22.legend(fontsize=30)
    
    plt.savefig(f'Output/{str}/Lat_Compare.png') 
    plt.close('all')
    
    fig_3 = plt.figure(figsize=(24,20))
    gs_3 = fig_3.add_gridspec(2, 2)
    
    dsigma_pred_select = list(map(lambda Wt: dsigma_New(Wt[0], -Wt[1], Ag0_bf, MAg_bf, Cg0_bf, MCg_bf, Aq0_bf, MAq_bf, Cq0_bf, MCq_bf, P_order = P_ORDER), zip(dsigmadata_select[:,0], dsigmadata_select[:,1])))
    dsigma_pred_all = list(map(lambda Wt: dsigma_New(Wt[0], -Wt[1], Ag0_bf, MAg_bf, Cg0_bf, MCg_bf, Aq0_bf, MAq_bf, Cq0_bf, MCq_bf, P_order = P_ORDER), zip(dsigmadata_reshape[:,0], dsigmadata_reshape[:,1])))
    
    ax11 = fig_3.add_subplot(gs_3[0, 0])
    ax11.errorbar(dsigmadata_select[:,1], dsigmadata_select[:,2], yerr = dsigmadata_select[:,3], fmt='o', capsize=5, capthick=1, ecolor='red', label="Xsec Data")
    ax11.scatter(dsigmadata_select[:,1], dsigma_pred_select, color='blue', marker='D', label='XSec Fit')
    ax11.legend(fontsize=30)

    ax12 = fig_3.add_subplot(gs_3[0, 1])
    ax12.errorbar(dsigmadata_select[:,1], dsigmadata_select[:,2], yerr = dsigmadata_select[:,3], fmt='o', capsize=5, capthick=1, ecolor='red', label="Xsec Data")
    ax12.scatter(dsigmadata_select[:,1], dsigma_pred_select, color='blue', marker='D', label='XSec Fit')
    ax12.legend(fontsize=30)
    ax12.set_yscale("log")
    
    ax21 = fig_3.add_subplot(gs_3[1, 0])
    ax21.errorbar(dsigmadata_reshape[:,1], dsigmadata_reshape[:,2], yerr = dsigmadata_reshape[:,3], fmt='o', capsize=5, capthick=1, ecolor='red', label="Xsec Data")
    ax21.scatter(dsigmadata_reshape[:,1], dsigma_pred_all, color='blue', marker='D', label='XSec Fit')
    ax21.legend(fontsize=30)
    
    ax22 = fig_3.add_subplot(gs_3[1, 1])
    ax22.errorbar(dsigmadata_reshape[:,1], dsigmadata_reshape[:,2], yerr = dsigmadata_reshape[:,3], fmt='o', capsize=5, capthick=1, ecolor='red', label="Xsec Data")
    ax22.scatter(dsigmadata_reshape[:,1], dsigma_pred_all, color='blue', marker='D', label='XSec Fit')
    ax22.legend(fontsize=30)
    ax22.set_yscale("log")
    
    plt.savefig(f'Output/{str}/Exp_Compare.png')
    plt.close('all')

INCLUDE_XSEC = False
P_ORDER = 1
fit("lattice_only")
print("Lattice only fit and plot finished...")

INCLUDE_XSEC = True
P_ORDER = 1
fit("lattice_LOexp")
print("Lattice + LO experimental data fit and plot finished...")

INCLUDE_XSEC = True
P_ORDER = 2
fit("lattice_NLOexp")
print("Lattice + NLO experimental data fit and plot finished...")
