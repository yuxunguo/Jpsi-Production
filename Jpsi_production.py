import numpy as np
from iminuit import Minuit
from scipy.integrate import quad

Mproton = 0.938
MJpsi = 3.097
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

"""
# Later MIT lattice result with tripole parameterizations

A0Lat = 0.429
MALat = 1.64
C0Lat = -1.93 /4
MCLat = 1.07

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

def G2(W: float, t: float, A0: float, MA: float, C0: float, MC: float): 
    return Xi(W ,t) ** (-4) * ((1- t/ (4 * Mproton ** 2))* FormFactors(t, C0, MC) ** 2 * (4 * Xi(W ,t) ** 2) ** 2 + 2* FormFactors(t, A0, MA) * FormFactors(t, C0, MC)*4 * Xi(W ,t) ** 2 + (1- Xi(W ,t) ** 2) * FormFactors(t,A0,MA) **2)

def dsigma(W: float, t: float, A0: float, MA: float, C0: float, MC: float):
    return 1/conv * alphaEM * (2/3) **2 /(4* (W ** 2 - Mproton ** 2) ** 2) * (16 * np.pi * alphaS)** 2/ (3 * MJpsi ** 3) * psi2 * G2(W, t, A0, MA, C0, MC)

def sigma(W: float, A0: float, MA: float, C0: float, MC: float):
    return quad(lambda u: dsigma(W, u, A0, MA, C0, MC), tmin(W), tmax(W))[0]

GlueXsigmaCSV = open("GlueX_Total_xsection.csv")
GlueXsigma = np.loadtxt(GlueXsigmaCSV, delimiter=",")
GlueXdsigmaCSV = open("GlueX_differential_xsection.csv")
GlueXdsigma = np.loadtxt(GlueXdsigmaCSV, delimiter=",")

def chi2(A0: float, MA: float, C0: float, MC: float):

    sigma_pred = list(map(lambda W: sigma(W, A0, MA, C0, MC), GlueXsigma[:,0]))
    chi2sigma = np.sum(((sigma_pred - GlueXsigma[:,2]) / GlueXsigma[:,3]) **2 )
    Wdsigma = 4.58
    dsigma_pred = list(map(lambda t: dsigma(Wdsigma, - t, A0, MA, C0, MC), GlueXdsigma[:,0]))
    chi2dsigma = np.sum(((dsigma_pred - GlueXdsigma[:,2]) / GlueXdsigma[:,3]) **2 )
    return chi2sigma + chi2dsigma

m = Minuit(chi2, A0 = A0Lat, MA = MALat, C0 = C0Lat ,MC = MCLat)
m.errordef = 1
m.fixed["A0"] = True
m.fixed["MC"] = True
m.migrad()
m.hesse()

print(m.values)

print(m.errors)

print(m.params)
