import numpy as np
from iminuit import Minuit

from math import gamma
from scipy.integrate import quad
from scipy.special import hyp2f1

Mproton = 0.938
MJpsi = 3.097
alphaEM = 1/133
alphaS = 0.30187

Norm = 2.032 

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

"""

def Kpsi(W :float):
    return np.sqrt(((W**2 -(MJpsi - Mproton)**2) *( W**2 -(MJpsi + Mproton)**2 ))/(4.*W**2))

def PCM(W: float):
    return np.sqrt(( W**2 - Mproton**2 )**2/(4.*W**2))

def tmin(W: float):
    return 2* Mproton ** 2 - 2 * np.sqrt((Mproton**2 + Kpsi(W)**2)*(Mproton**2 + PCM(W)**2)) - 2 * Kpsi(W) * PCM(W)

def tmax(W: float):
    return 2* Mproton ** 2 - 2 * np.sqrt((Mproton**2 + Kpsi(W)**2)*(Mproton**2 + PCM(W)**2)) + 2 * Kpsi(W) * PCM(W)

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
    return np.sqrt(Mproton**2 + 2 * E_gamma * Mproton)

def E_from_W(W: float):
    return (W ** 2 -Mproton ** 2)/ (2 * Mproton)

def Eta(W: float, t: float):
    return MJpsi ** 2/(4 * Mproton * E_from_W(W) - MJpsi **2 + t)

def dsigma(W: float, t: float, A0: float, MA: float, C0: float, MC: float):
    eta = Eta(W, t)
    return  Norm / (64 * np.pi * (W ** 2 - Mproton ** 2 ) **2) * (FormFactors(t, A0, MA) + eta **2 / 4 * FormFactors(t, C0, MC)) ** 2/ (A0 ** 2) * (Mproton * E_from_W(W))**4 * 8

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

print(chi2(A0Lat,MALat,C0Lat,MCLat)/17)

m = Minuit(chi2, A0 = A0Lat, MA = MALat, C0 = 0 ,MC = MCLat)
m.errordef = 1
#m.fixed["A0"] = True
m.fixed["C0"] = True
m.fixed["MC"] = True
m.migrad()
m.hesse()

print(m.values)

print(m.errors)

print(m.params)
