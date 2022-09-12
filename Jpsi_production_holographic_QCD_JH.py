import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iminuit import Minuit
import uncertainties
from uncertainties import ufloat
from uncertainties import unumpy

power = 3

Mproton = 0.938
MJpsi = 3.097
alphaEM = 1/133
alphaS = 0.30187


# Later MIT lattice result with tripole parameterizations

A0Lat = 0.429
MALat = 1.64
C0Lat = -1.93 /4
MCLat = 1.07

A0CT18 = 0.414
u_A0CT18 = ufloat(0.414, 0.008)

def FormFactors(t: float, A0: float, Mpole: float):
    return A0/(1 - t / (Mpole ** 2)) ** 3

"""

# Earlier MIT lattice result with dipole parameterizations

A0Lat = 0.58
MALat = 1.13
C0Lat = -1
MCLat = 0.48

def FormFactors(t: float, A0: float, Mpole: float):
    return A0/(1 - t / (Mpole ** 2)) ** 2

"""

def Kpsi(W :float):
    return unumpy.sqrt(((W**2 -(MJpsi - Mproton)**2) *( W**2 -(MJpsi + Mproton)**2 ))/(4.*W**2))

def PCM(W: float):
    return unumpy.sqrt(( W**2 - Mproton**2 )**2/(4.*W**2))

def tmin(W: float):
    return 2* Mproton ** 2 - 2 * unumpy.sqrt((Mproton**2 + Kpsi(W)**2)*(Mproton**2 + PCM(W)**2)) - 2 * Kpsi(W) * PCM(W)

def tmax(W: float):
    return 2* Mproton ** 2 - 2 * unumpy.sqrt((Mproton**2 + Kpsi(W)**2)*(Mproton**2 + PCM(W)**2)) + 2 * Kpsi(W) * PCM(W)

def PPlus(W: float):
    return W/unumpy.sqrt(2)

def PprimePlus(W: float, t: float):
    return unumpy.sqrt(Mproton**2 + Kpsi(W)**2)/unumpy.sqrt(2) + (-2*Mproton**2 + t + 2*unumpy.sqrt((Mproton**2 + Kpsi(W)**2)*(Mproton**2 + PCM(W)**2)))/(2.*unumpy.sqrt(2)*PCM(W))

def Xi(W: float, t: float):
    return (PPlus(W) - PprimePlus(W,t))/(PPlus(W) + PprimePlus(W,t))

"""
def AFF(t: float, A0:float, kappaT: float):
    K2 = -t
    aK = K2 / (4 * kappaT ** 2)
    return A0 * 6 * gamma(2 + aK/2)/gamma(4 + aK/2) * hyp2f1(3, aK/2, aK/2+4, -1)

def ASFF(t: float, AS0:float, kappaS: float):
    K2 = -t
    atK = K2 / (4 * kappaS ** 2)
    return AS0 * 6 * gamma(2 + atK/2)/gamma(4 + atK/2) * hyp2f1(3, atK/2, atK/2+4, -1)

def DFF(t: float, A0:float, kappaT: float, AS0:float, kappaS: float):
    return 4/3 * Mproton ** 2 / t  (AFF(t, A0, kappaT) - ASFF(t, AS0, kappaS))
"""

def W_from_E(E_gamma: float):
    return unumpy.sqrt(Mproton**2 + 2 * E_gamma * Mproton)

def E_from_W(W: float):
    return (W ** 2 -Mproton ** 2)/ (2 * Mproton)

def Eta(W: float, t: float):
    return MJpsi ** 2/(4 * Mproton * E_from_W(W) - MJpsi **2 + t)

def dsigma(W: float, t: float, A0: float, MA: float, C0: float, MC: float, Norm: float):
    eta = Eta(W, t)
    return  Norm / (64 * np.pi * (W ** 2 - Mproton ** 2 ) **2) * (FormFactors(t, A0, MA) + eta **2 * 4 * FormFactors(t, C0, MC)) ** 2/ (A0 ** 2) * (Mproton * E_from_W(W))**4 * 8 #* (1 - t/ (4 * Mproton **2))

#def sigma(W: float, A0: float, MA: float, C0: float, MC: float):
#    return quad(lambda u: dsigma(W, u, A0, MA, C0, MC), tmin(W), tmax(W))[0]


data = pd.read_excel("2022-final-xsec-electron-channel.xlsx")
data_E = data['avg_E'] 
data_W = W_from_E(data_E)
data_t =  - data['avg_abs_t']
data_Xi = Xi(data_W, data_t)
data_dsigma = data['dsdt_nb']
data_dsigma_err = data['tot_error']

data_E_err = (data['E_high'] - data['E_low'])/2 / np.sqrt(3)
u_data_E = unumpy.uarray(data_E, data_E_err)
u_data_W = W_from_E(u_data_E)
data_t_err = (data['abst_t_tmin_high'] - data['abs_t_tmin_low'])/2 / np.sqrt(3)
u_data_t = unumpy.uarray(data_t, data_t_err)
u_data_Xi = Xi(u_data_W, u_data_t)

def chi2(A0:float, MA: float, C0: float, MC: float, Norm: float):

    dsigma_pred = dsigma(data_W, data_t, A0, MA, C0, MC, Norm)
    chi2dsigma = np.sum( ( (dsigma_pred - data_dsigma)/data_dsigma_err )**2 )
   
    return chi2dsigma

m = Minuit(chi2, A0=A0CT18, MA = 2.6, C0 = -0.2 ,MC = 1.2, Norm = 8.5)
m.errordef = 1
m.fixed["A0"] = True
m.fixed["Norm"] = True
m.migrad()
m.hesse()


print(m.params)


u_MA_paper = ufloat(2.71, 0.19)
u_C0_paper = ufloat(-0.20, 0.11)
u_MC_paper = ufloat(1.28, 0.50)

u_MA_fit = ufloat(m.values[1], m.errors[1])
u_C0_fit = ufloat(m.values[2], m.errors[2])
u_MC_fit = ufloat(m.values[3], m.errors[3])
u_Norm_fit = ufloat(m.values[4], m.errors[4])

print(chi2(m.values[0],m.values[1],m.values[2],m.values[3],m.values[4]))

# Pick an energy bin

E_idx = 8 # this can be any number from 1 to 12; this selects a E gamma range


filt = data['E_idx'] == E_idx
bin_E_low = data['E_low'][filt].to_numpy()[0]
bin_E_high = data['E_high'][filt].to_numpy()[0]
bin_E_avg = data['avg_E'][filt].to_numpy()[0]

u_bin_E = u_data_E[filt][0]
u_bin_W = W_from_E(u_bin_E)

fig, ax = plt.subplots()
ax.set(yscale='log', xlabel='-t(GeV)', ylabel=r'd$\sigma$/d$t$(nb/GeV$^2$)')
ax.errorbar(np.abs(data_t[filt]), data_dsigma[filt], yerr = data_dsigma_err[filt], fmt='.', c='k')

xs = np.linspace(0.5, 4.5)

u_bin_dsigma_fit = dsigma(u_bin_W, -xs, u_A0CT18, u_MA_fit, u_C0_fit, u_MC_fit,u_Norm_fit)
bin_dsigma_fit = unumpy.nominal_values(u_bin_dsigma_fit)
bin_dsigma_fit_err = unumpy.std_devs(u_bin_dsigma_fit)

ax.plot(xs, bin_dsigma_fit, label='fit')
ax.fill_between(xs, bin_dsigma_fit - bin_dsigma_fit_err, bin_dsigma_fit + bin_dsigma_fit_err, alpha=0.2)
ax.text(0.73, 0.95, f'E in [{bin_E_low}, {bin_E_high}]', transform=ax.transAxes)

plt.show()