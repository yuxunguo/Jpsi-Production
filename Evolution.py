"""
NLO QCD evolution of moment space GPD. Many credits to K. Kumericki at https://github.com/kkumer/gepard.

Note:
    Functions in this module have as first argument Mellin moment
    n = j + 1, where j is conformal moment used everywhere else.
    Thus, they should always be called as f(j+1, ...).

"""
# from cmath import exp
# from scipy.special import loggamma as clngamma
#from this import d

import numpy as np
from scipy.special import psi, zeta, gamma, orthogonal, loggamma
from math import factorial, log
from typing import Tuple, Union
from numba import vectorize, njit

"""
***********************QCD constants***************************************
Refer to the constants.py at https://github.com/kkumer/gepard.
"""

M_jpsi = 3.097
NC = 3
CF = (NC**2 - 1) / (2 * NC)
CA = NC
CG = CF - CA/2
TF = 0.5
Alpha_Ref = 0.305
# All unit in GeV for dimensional quantities.
Ref_Scale = 2
# One loop accuracy for running strong coupling constant. 
nloop_alphaS = 2
# Initial scale of distribution functions at mc
Init_Scale_Q = M_jpsi/2

# Transform the original flavor basis to the evolution basis (qVal, q_du_plus, q_du_minus, qSigma, g)
# The same basis for PDF evolution are used, check references.

flav_trans =np.array([[1, 0, 1, 0, 0],
                     [-1, -2, 1, 2, 0],
                     [-1, 0, 1, 0 , 0],
                     [1, 2, 1, 2, 0],
                     [0, 0, 0, 0, 1]])

inv_flav_trans = np.linalg.inv(flav_trans)

f_rho_u = 0.209 # Change to 0.222
f_rho_d = 0.209 # Change to 0.210
f_rho_g = 0.209 # Change to 0.216
f_phi = 0.221 # Change to 0.233
f_jpsi = 0.406
'''
TFF_rho_trans = np.array([f_rho_u * 2 / 3 / np.sqrt(2), f_rho_u * 4 / 3 / np.sqrt(2), f_rho_d * 1 / 3 / np.sqrt(2), f_rho_d * 2 / 3 / np.sqrt(2), f_rho_g * 3 / 4 / np.sqrt(2)])#np.array([f_rho_u * 2 / 3 / np.sqrt(2), f_rho_u * 4 / 3 / np.sqrt(2), f_rho_d / 3 / np.sqrt(2), f_rho_d * 2 / 3 / np.sqrt(2), f_rho_g * 3 / 4 / np.sqrt(2)])
TFF_phi_trans = np.array([0, 0, 0, 0, -1/3]) # strange contribution should be included but doesn't exist in current 2 quark framework
TFF_jpsi_trans = np.array([0, 0, 0, 0, 2/3])
'''
"""
***********************pQCD running coupling constant***********************
Here rundec is used instead.
"""

B00 = 11./3. * CA
B01 = -4./3. * TF
B10 = 34./3. * CA**2
B11 = -20./3. * CA*TF - 4. * CF*TF

@njit(["float64(int32)"])
def beta0(nf: int) -> float:
    """ LO beta function of pQCD, will be used for LO GPD evolution. """
    return - B00 - B01 * nf

@njit(["float64(int32)"])
def beta1(nf):
    """ NLO beta function of pQCD """
    return - B10 - B11 * nf

@njit(["float64[:](float64[:], int32)", "float64(float64, int32)"])
def _fbeta1(a: float, nf: int) -> float:
    return a**2 * (beta0(nf) + a * beta1(nf))

@njit(["float64[:](int32, float64[:])", "float64(int32, float64)"])
def AlphaS0(nf: int, Q: float) -> float:
    return Alpha_Ref / (1 - Alpha_Ref/2/np.pi * beta0(nf) * np.log(Q/Ref_Scale))

@njit(["float64[:](int32, float64[:])", "float64(int32, float64)"])
def AlphaS1(nf: int, Q: float) -> float:
    NASTPS = 40
    
    # a below is as defined in 1/4pi expansion
    a = np.ones_like(Q) * Alpha_Ref / 4 / np.pi
    lrrat = 2 * np.log(Q/Ref_Scale)
    dlr = lrrat / NASTPS

   
    for k in range(1, NASTPS+1):
        xk0 = dlr * _fbeta1(a, nf)
        xk1 = dlr * _fbeta1(a + 0.5 * xk0, nf)
        xk2 = dlr * _fbeta1(a + 0.5 * xk1, nf)
        xk3 = dlr * _fbeta1(a + xk2, nf)
        a += (xk0 + 2 * xk1 + 2 * xk2 + xk3) / 6


    # Return to .../(2pi)  expansion
    a *= 4*np.pi
    return a

@njit(["float64[:](int32, int32, float64[:])", "float64(int32, int32, float64)"])
def AlphaS(nloop: int, nf: int, Q: float) -> float:
    if nloop==1:
        return AlphaS0(nf, Q)
    if nloop==2:
        return AlphaS1(nf, Q)
    raise ValueError('Only LO and NLO implemented!')


"""
***********************Anomalous dimensions of GPD in the moment space*****
Refer to the adim.py at https://github.com/kkumer/gepard.
"""

# Fixed quad function that allow more general function. The func here take input of shape (N,) and output (N,......) which doesn't have to be (N,)
def fixed_quadvec(func, a, b, n=100, args=()):
    rootsNLO, weightsNLO = orthogonal.p_roots(n)
    y = (b-a) * (rootsNLO + 1)/2.0 + a
    yfunc = func(y)
    return (b-a)/2.0*np.einsum('j,j...->...',weightsNLO,yfunc)
    
def pochhammer(z: Union[complex, np.ndarray], m: int) -> Union[complex, np.ndarray]:
    """Pochhammer symbol.

    Args:
        z: complex argument
        m: integer index

    Returns:
        complex: pochhammer(z,m)

    """
    p = z
    for k in range(1, m):
        p = p * (z + k)
    return p

poch = pochhammer  # just an abbreviation

def dpsi_one(z: complex, m: int) -> complex:
    """Polygamma - m'th derivative of Euler gamma at z."""
    # Algorithm from Vogt, cf. julia's implementation
    sub = 0j

    if z.imag < 10:
        subm = (-1/z)**(m+1) * factorial(m)
        while z.real < 10:
            sub += subm
            z += 1
            subm = (-1/z)**(m+1) * factorial(m)

    a1 = 1.
    a2 = 1./2.
    a3 = 1./6.
    a4 = -1./30.
    a5 = 1./42.
    a6 = -1./30.
    a7 = 5./66.

    if m != 1:
        for k2 in range(2, m+1):
            a1 = a1 * (k2-1)
            a2 = a2 * k2
            a3 = a3 * (k2+1)
            a4 = a4 * (k2+3)
            a5 = a5 * (k2+5)
            a6 = a6 * (k2+7)
            a7 = a7 * (k2+9)

    rz = 1. / z
    dz = rz * rz
    res = (sub + (-1)**(m+1) * rz**m *
           (a1 + rz * (a2 + rz * (a3 + dz *
            (a4 + dz * (a5 + dz * (a6 + a7 * dz)))))))
    return res

dpsi = np.vectorize(dpsi_one)

def S1(z: Union[complex, np.ndarray]) -> Union[complex, np.ndarray]:
    """Harmonic sum S_1."""
    return np.euler_gamma + psi(z+1)

def S2(z: Union[complex, np.ndarray]) -> Union[complex, np.ndarray]:
    """Harmonic sum S_2."""
    return zeta(2) - dpsi(z+1, 1)

def S3(z: Union[complex, np.ndarray]) -> Union[complex, np.ndarray]:
    """Harmonic sum S_3."""
    return zeta(3) + dpsi(z+1, 2) / 2

def S4(z: Union[complex, np.ndarray]) -> Union[complex, np.ndarray]:
    """Harmonic sum S_4."""
    return zeta(4) - dpsi(z+1, 3) / 6

def S2_prime(z: Union[complex, np.ndarray], prty: int) -> Union[complex, np.ndarray]:
    """Curci et al Eq. (5.25)."""
    # note this is related to delS2
    return (1+prty)*S2(z)/2 + (1-prty)*S2(z-1/2)/2

def S3_prime(z: Union[complex, np.ndarray], prty: int) -> Union[complex, np.ndarray]:
    """Curci et al Eq. (5.25)."""
    return (1+prty)*S3(z)/2 + (1-prty)*S3(z-1/2)/2

def delS2(z: Union[complex, np.ndarray]) -> Union[complex, np.ndarray]:
    """Return Harmonic sum S_2 difference.

    Args:
        z: complex argument

    Returns:
        delS2((z+1)/2) From Eq. (4.13) of 1310.5394.
        Note halving of the argument.

    """
    return S2(z) - S2(z - 1/2)

def deldelS2(j: Union[complex, np.ndarray], k: int) -> Union[complex, np.ndarray]:
    """Return diference of harmonic sum S_2 differences.

    Args:
        j: complex argument
        k: integer index

    Returns:
        Equal to delS2((j+1)/2, (k+1)/2) From Eq. (4.38) of 1310.5394.
        Note halving of the argument.

    """
    return (delS2(j) - delS2(k)) / (4*(j-k)*(2*j+2*k+1))

def Sm1(z: Union[complex, np.ndarray], k: int) -> Union[complex, np.ndarray]:
    """Aux fun. FIXME: not tested."""
    return - log(2) + 0.5 * (1-2*(k % 2)) * (psi((z+2)/2) - psi((z+1)/2))

def MellinF2(n: Union[complex, np.ndarray]) -> Union[complex, np.ndarray]:
    """Return Mellin transform  i.e. x^(N-1) moment of Li2(x)/(1+x).

    Args:
        n: complex argument

    Returns:
        According to Eq. (33) in Bluemlein and Kurth, hep-ph/9708388

    """
    abk = np.array([0.9999964239, -0.4998741238,
                    0.3317990258, -0.2407338084, 0.1676540711,
                   -0.0953293897, 0.0360884937, -0.0064535442])
    psitmp = psi(n)
    mf2 = 0

    for k in range(1, 9):
        psitmp = psitmp + 1 / (n + k - 1)
        mf2 += (abk[k-1] *
                ((n - 1) * (zeta(2) / (n + k - 1) -
                 (psitmp + np.euler_gamma) / (n + k - 1)**2) +
                 (psitmp + np.euler_gamma) / (n + k - 1)))

    return zeta(2) * log(2) - mf2

def SB3(j: Union[complex, np.ndarray]) -> Union[complex, np.ndarray]:
    """Eq. (4.44e) of arXiv:1310.5394."""
    return 0.5*S1(j)*(-S2(-0.5+0.5*j)+S2(0.5*j))+0.125*(-S3(
             - 0.5 + 0.5 * j) + S3(0.5 * j)) - 2 * (0.8224670334241131 * (
                 -S1(0.5 * (-1 + j)) + S1(0.5 * j)) - MellinF2(1 + j))

def S2_tilde(n: Union[complex, np.ndarray], prty: int) -> Union[complex, np.ndarray]:
    """Eq. (30) of  Bluemlein and Kurth, hep-ph/9708388."""
    G = psi((n+1)/2) - psi(n/2)
    return -(5/8)*zeta(3) + prty*(S1(n)/n**2 - (zeta(2)/2)*G + MellinF2(n))

def lsum(m: Union[complex, np.ndarray], n: Union[complex, np.ndarray])-> Union[complex, np.ndarray]:
    
    return sum( (2*l+1)*(-1)**l * deldelS2((m+1)/2,l/2)/2 for l in range(1))

def lsumrev(m: Union[complex, np.ndarray], n: Union[complex, np.ndarray])-> Union[complex, np.ndarray]:
    
    return sum((2*l+1)*deldelS2((m+1)/2,l/2)/2 for l in range(1))

def non_singlet_LO(n:Union[complex, np.ndarray], nf: int, p: int, prty: int = 1) -> Union[complex, np.ndarray]:
    """Non-singlet LO anomalous dimension.

    Args:
        n (complex): which moment (= Mellin moment for integer n)
        nf (int): number of active quark flavors
        p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et)
        prty (int): 1 for NS^{+}, -1 for NS^{-}, irrelevant at LO

    Returns:
        Non-singlet LO anomalous dimension.
        
    It's an algebric equation, any shape of n should be fine.
    """
    return CF*(-3.0-2.0/(n*(1.0+n))+4.0*S1(n))

def singlet_LO(n: Union[complex, np.ndarray], nf: int, p: int, prty: int = 1) -> np.ndarray:
    """Singlet LO anomalous dimensions.

    Args:
        n (complex): which moment (= Mellin moment for integer n)
        nf (int): number of active quark flavors
        p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et)
        prty (int): C parity, irrelevant at LO

    Returns:
        2x2 complex matrix ((QQ, QG),
                            (GQ, GG))

    This will work as long as n, nf, and p can be broadcasted together.

    """

    '''
    if(p == 1):
        qq0 = CF*(-3.0-2.0/(n*(1.0+n))+4.0*S1(n))
        qg0 = (-4.0*nf*TF*(2.0+n+n*n))/(n*(1.0+n)*(2.0+n))
        gq0 = (-2.0*CF*(2.0+n+n*n))/((-1.0+n)*n*(1.0+n))
        gg0 = -4.0*CA*(1/((-1.0+n)*n)+1/((1.0+n)*(2.0+n))-S1(n)) -11*CA/3. + 4*nf*TF/3.

        return np.array([[qq0, qg0],
                        [gq0, gg0]])
    
    if(p == -1):
        qq0 = CF*(-3.0-2.0/(n*(1.0+n))+4.0*S1(n))
        qg0 = (-4.0*nf*TF*(-1.0+n))/(n*(1.0+n))
        gq0 = (-2.0*CF*(2.0+n))/(n*(1.0+n))
        gg0 = -4.0*CA*(2/(n*(1.0+n))-S1(n)) -11*CA/3. + 4*nf*TF/3.

        return np.array([[qq0, qg0],
                        [gq0, gg0]])
    '''

    epsilon = 0.00001 * ( n == 1)

    # Here, I am making the assumption that a is either 1 or -1
    qq0 = np.where(p>0,  CF*(-3.0-2.0/(n*(1.0+n))+4.0*S1(n)),           CF*(-3.0-2.0/(n*(1.0+n))+4.0*S1(n)))
    qg0 = np.where(p>0,  (-4.0*nf*TF*(2.0+n+n*n))/(n*(1.0+n)*(2.0+n)),  (-4.0*nf*TF*(-1.0+n))/(n*(1.0+n)) )
    gq0 = np.where(p>0,  (-2.0*CF*(2.0+n+n*n))/((-1.0+n + epsilon)*n*(1.0+n)),    (-2.0*CF*(2.0+n))/(n*(1.0+n)))
    gg0 = np.where(p>0,  -4.0*CA*(1/((-1.0+n + epsilon)*n)+1/((1.0+n)*(2.0+n))-S1(n)) -11*CA/3. + 4*nf*TF/3., \
        -4.0*CA*(2/(n*(1.0+n))-S1(n)) -11*CA/3. + 4*nf*TF/3. )

    # all of the four above have shape (N)
    # more generally, if p is a multi dimensional array, like (N1, N1, N2)... Then this could also work

    qq0_qg0 = np.stack((qq0, qg0), axis=-1)
    gq0_gg0 = np.stack((gq0, gg0), axis=-1)

    return np.stack((qq0_qg0, gq0_gg0), axis=-2)# (N, 2, 2)

def non_singlet_NLO(n: complex, nf: int, prty: int) -> complex:
    """Non-singlet anomalous dimension.

    Args:
        n (complex): which moment (= Mellin moment for integer n)
        nf (int): number of active quark flavors
        prty (int): 1 for NS^{+}, -1 for NS^{-}

    Returns:
        Non-singlet NLO anomalous dimension.
        
    This will work as long as n, nf, and prty can be broadcasted together.
    """
    # From Curci et al.
    nlo = (CF * CG * (
            16*S1(n)*(2*n+1)/poch(n, 2)**2 +
            16*(2*S1(n) - 1/poch(n, 2)) * (S2(n)-S2_prime(n/2, prty)) +
            64 * S2_tilde(n, prty) + 24*S2(n) - 3 - 8*S3_prime(n/2, prty) -
            8*(3*n**3 + n**2 - 1)/poch(n, 2)**3 -
            16*prty*(2*n**2 + 2*n + 1)/poch(n, 2)**3) +
           CF * CA * (S1(n)*(536/9 + 8*(2*n+1)/poch(n, 2)**2) - 16*S1(n)*S2(n) +
                      S2(n)*(-52/3 + 8/poch(n, 2)) - 43/6 -
                      4*(151*n**4 + 263*n**3 + 97*n**2 + 3*n + 9)/9/poch(n, 2)**3) +
           CF * nf * TF * (-(160/9)*S1(n) + (32/3)*S2(n) + 4/3 +
                           16*(11*n**2 + 5*n - 3)/9/poch(n, 2)**2)) / 4

    return nlo
    

def singlet_NLO(n: complex, nf: int, p: int, prty: int = 1) -> np.ndarray:
    """Singlet NLO anomalous dimensions matrix.
    
    Args:
        n (complex): which moment (= Mellin moment for integer n)
        nf (int): number of active quark flavors
        p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et)
        prty (int): C parity
            
    Returns:
        Matrix (LO, NLO) where each is in turn
        2x2 complex matrix
        ((QQ, QG),
        (GQ, GG))
        
    """
    
   # qq1 = non_singlet_NLO(n, nf, 1) + np.ones_like(p) * (-4*CF*TF*nf*((2 + n*(n + 5))*(4 + n*(4 + n*(7 + 5*n)))/(n-1)/n**3/(n+1)**3/(n+2)**2))
    
   # qg1 = np.ones_like(p) * 2*nf*TF*(S1(n)*(2*S1(n)*(-CF*(n-1)*(n+2)**2*(2+n+n**2)*n**2*(n+1)**2 + CA*n**2*(n+1)**2*(n+2)**2*(n**3 + n - 2)) - CF*(n-1)*(n+2)**2*(-4*n*(n+1)**3*(n+2)) - 2*CA*(4*(n-1)*n**3*(n+1)*(n+2)*(2*n + 3))) + S2_prime(n)*(2*CA*n**2*(n+1)**2*(n+2)**2*(n**3 + n - 2)) - CF*(n-1)*(n+1)**2*(64 + 160*(n-1) + 159*(n-1)**2 + 70*(n-1)**3 + 11*(n-1)**4 + n**2*(n+1)**2*5*(2 + n + n**2)) - 2*CA*(16 + 64*n + 104*n**2 + 128*n**3 + 85*n**4 + 36*n**5 + 25*n**6 + 15*n**7 + 6*n**8 + n**9))/(n-1)/n**3/(n+1)**3/(n+2)**3
    
  #  gq1 = np.ones_like(p) * (-CF)*(S1(n)*(S1(n)*(18*CA*n**2*(n+1)**2*(n+2)**2*(n**3 + n - 2) - 18*CF*(n-1)*n**2*(n+2)**2*(2 + 5*n + 5*n**2 + 3*n**3 + n**4)) + 18*CF*(n-1)*n**2*(n+2)**2*(10 + 27*n + 25*n**2 + 13*n**3 + 5*n**4) - 6*CA*n*(n+1)**2*(n+2)**2*(-12 + n*(-22 + 41*n + 17*n**3)) + 24*(n-1)*n**2*(n+1)**2*(n+2)**2*(n**2 + n + 2)*TF*nf) + S2(n)*(18*CA*n**2*(n+1)**2*(n+2)**2*(n**3 + n - 2) - 18*CF*(n-1)*n**2*(n+2)**2*(2 + 5*n + 5*n**2 + 3*n**3 + n**4)) + 9*CF*(n-1)*(n+2)**2*(-4 + n*(-12 + n*(-1 + 28*n + 43*n + 43*n**2 + 30*n**3 + 12*n**4))) + 2*CA*(2592 + 21384*(n-1) + 72582*(n-1)**2 + 128024*(n-1)**3 + 133818*(n-1)**4 + 88673*(n-1)**5 + 38822*(n-1)**6 + 10292*(n-1)**7 + 1662*(n-1)**8 + 109*(n-1)**9 - 9*n**2*(n+1)**2*(n+2)**2*(n**3 + n - 2)*S2_prime(n)) - 8*(n-1)*n**2*(n+1)**2*(n+2)**2*(16 + 27*n + 13*n**2 + 8*n**3)*TF*nf)/9/n**3/(n+1)**3/(n**2 + n - 2)**2
    
  #  gg1 = np.ones_like(p) * (S1(n)*(CA**2*(134/9 + 16*(2*n + 1)*(-2 + n*(n+1)*(n**2 + n + 2))/(n-1)**2*n**2*(n+1)**2*(n+2)**2 - 4*S2_prime(n)) - 40*CA*TF*nf/9) + CA**2*(-16/3 - (576 + n*(1488 + n*(560 + n*(-1632 + n*(-2344 + n*(1567 + n*(6098 + n*(6040 + 457*n*(6 + n)))))))))/9/(n-1)**2/n**3/(n+1)**3/(n+2)**3 + 8*(n*82 + n + 1)*S2_prime(n)/(n-1)*n*(n+1)*(n+2) - S3_prime(n) + 8*S2_tilde(n)) + 8*CF*TF*nf*(3 + (6+ n*(n+1)*(28 + 19*n*(n+1)))/n**2/(n+1)**2/(n**2 + n - 2))/9 + 2*CF*TF*nf*(-8 + n*(-8 + n*(-10 + n*(-22 + n*(-3 + n*(6 + n*(8 + n*(4 + n)))))))))
    
  #  qq1_qg1 = np.stack((qq1, qg1), axis=-1)
  #  gq1_gg1 = np.stack((gq1, gg1), axis=-1)

   # return np.stack((qq1_qg1, gq1_gg1), axis=-2)# (N, 2, 2)
    
    
   # qq1 = np.where(p>0, non_singlet_NLO(n, nf, 1) - 4*CF*TF*nf*(5*n**5+32*n**4+49*n**3+38*n**2 + 28*n+8)/((n-1)*n**3*(n+1)**3*(n+2)**2), non_singlet_NLO(n, nf, 1) - 4*CF*TF*nf*(5*n**5+32*n**4+49*n**3+38*n**2 + 28*n+8)/((n-1)*n**3*(n+1)**3*(n+2)**2))

   # qg1 = np.where(p>0,(-8*CF*nf*TF*((-4*S1(n))/n**2+(4+8*n + 26*n**3 + 11*n**4 + 15*(n*n))/(n**3*(1+n)**3*(2+n)) + ((2+n+n*n)*(5-2*S2(n) + 2*(S1(n)*S1(n))))/(n*(1+n)*(2+n))) - 8*CA*nf*TF*((8*(3+2*n)*S1(n))/((1+n)**2*(2+n)**2) + (2*(16+64*n+128*n**3+85*n**4+36*n**5+25*n**6 + 15*n**7+6*n**8+n**9+104*(n*n)))/( (-1+n)*n**3*(1+n)**3*(2+n)**3)+( (2+n+n*n)*(2*S2(n)-2*(S1(n)*S1(n))-2*S2(n/2)))/(n*(1+n)*(2+n))))/4, (-8*CF*nf*TF*((-4*S1(n))/n**2+(4+8*n + 26*n**3 + 11*n**4 + 15*(n*n))/(n**3*(1+n)**3*(2+n)) + ((2+n+n*n)*(5-2*S2(n) + 2*(S1(n)*S1(n))))/(n*(1+n)*(2+n))) - 8*CA*nf*TF*((8*(3+2*n)*S1(n))/((1+n)**2*(2+n)**2) + (2*(16+64*n+128*n**3+85*n**4+36*n**5+25*n**6 + 15*n**7+6*n**8+n**9+104*(n*n)))/( (-1+n)*n**3*(1+n)**3*(2+n)**3)+( (2+n+n*n)*(2*S2(n)-2*(S1(n)*S1(n))-2*S2(n/2)))/(n*(1+n)*(2+n))))/4)

   # gq1 = np.where(p>0, (-(32/3)*CF*nf*TF*((1+n)**(-2) + ((-(8/3)+S1(n))*(2+n+n*n))/((-1+n)*n*(1+n))) - 4*(CF*CF)*((-4*S1(n))/(1+n)**2-( -4-12*n+28*n**3+43*n**4 + 30*n**5+12*n**6-n*n)/((-1+n)*n**3*(1+n)**3) + ((2+n+n*n)*(10*S1(n)-2*S2(n)-2*(S1(n)*S1(n))))/((-1+n)*n*(1+n))) - 8*CF*CA*(((1/9)*(144+432*n-1304*n**3-1031*n**4 + 695*n**5+1678*n**6+1400*n**7+621*n**8+109*n**9 - 152*(n*n)))/((-1+n)**2*n**3*(1+n)**3*(2+n)**2) - ((1/3)*S1(n)*(-12-22*n+17*n**4 + 41*(n*n)))/((-1+n)**2*n**2*(1+n))+( (2+n+n*n)*(S2(n) + S1(n)*S1(n)-S2(n/2)))/((-1+n)*n*(1+n))))/4, (-(32/3)*CF*nf*TF*((1+n)**(-2) + ((-(8/3)+S1(n))*(2+n+n*n))/((-1+n)*n*(1+n))) - 4*(CF*CF)*((-4*S1(n))/(1+n)**2-( -4-12*n+28*n**3+43*n**4 + 30*n**5+12*n**6-n*n)/((-1+n)*n**3*(1+n)**3) + ((2+n+n*n)*(10*S1(n)-2*S2(n)-2*(S1(n)*S1(n))))/((-1+n)*n*(1+n))) - 8*CF*CA*(((1/9)*(144+432*n-1304*n**3-1031*n**4 + 695*n**5+1678*n**6+1400*n**7+621*n**8+109*n**9 - 152*(n*n)))/((-1+n)**2*n**3*(1+n)**3*(2+n)**2) - ((1/3)*S1(n)*(-12-22*n+17*n**4 + 41*(n*n)))/((-1+n)**2*n**2*(1+n))+( (2+n+n*n)*(S2(n) + S1(n)*S1(n)-S2(n/2)))/((-1+n)*n*(1+n))))/4)

   # gg1 = np.where(p>0, (CF*nf*TF*(8+(16*(-4-4*n-10*n**3+n**4+4*n**5+2*n**6 - 5*(n*n)))/((-1+n)*n**3*(1+n)**3*(2+n))) + CA*nf*TF*(32/3 - (160/9)*S1(n)+( (16/9)*(12+56*n+76*n**3+38*n**4+94*(n*n)))/((-1+n)*n**2*(1+n)**2*(2+n))) + CA*CA*(-64/3+(536/9)*S1(n)+(64*S1(n)*( -2-2*n+8*n**3+5*n**4+2*n**5+7*(n*n)))/((-1+n)**2*n**2*(1+n)**2*(2+n)**2) - ((4/9)*(576+1488*n-1632*n**3-2344*n**4+1567*n**5 + 6098*n**6+6040*n**7+2742*n**8+457*n**9+560*(n*n)))/( (-1+n)**2*n**3*(1+n)**3*(2+n)**3) - 16*S1(n)*S2(n/2)+(32*(1+n+n*n)*S2(n/2))/( (-1+n)*n*(1+n)*(2+n))-4*S3(n/2) + 32*(S1(n)/n**2-(5/8)*zeta(3)+MellinF2(n) - zeta(2)*(-psi(n/2)+psi((1+n)/2))/2)))/4, (CF*nf*TF*(8+(16*(-4-4*n-10*n**3+n**4+4*n**5+2*n**6 - 5*(n*n)))/((-1+n)*n**3*(1+n)**3*(2+n))) + CA*nf*TF*(32/3 - (160/9)*S1(n)+( (16/9)*(12+56*n+76*n**3+38*n**4+94*(n*n)))/((-1+n)*n**2*(1+n)**2*(2+n))) + CA*CA*(-64/3+(536/9)*S1(n)+(64*S1(n)*( -2-2*n+8*n**3+5*n**4+2*n**5+7*(n*n)))/((-1+n)**2*n**2*(1+n)**2*(2+n)**2) - ((4/9)*(576+1488*n-1632*n**3-2344*n**4+1567*n**5 + 6098*n**6+6040*n**7+2742*n**8+457*n**9+560*(n*n)))/( (-1+n)**2*n**3*(1+n)**3*(2+n)**3) - 16*S1(n)*S2(n/2)+(32*(1+n+n*n)*S2(n/2))/( (-1+n)*n*(1+n)*(2+n))-4*S3(n/2) + 32*(S1(n)/n**2-(5/8)*zeta(3)+MellinF2(n) - zeta(2)*(-psi(n/2)+psi((1+n)/2))/2)))/4)


    qq1 = non_singlet_NLO(n, nf, 1) + np.ones_like(p) * (-4*CF*TF*nf*(5*n**5+32*n**4+49*n**3+38*n**2 + 28*n+8)/((n-1)*n**3*(n+1)**3*(n+2)**2))
    
    qg1 = np.ones_like(p) * ((-8*CF*nf*TF*((-4*S1(n))/n**2+(4+8*n + 26*n**3 + 11*n**4 + 15*(n*n))/(n**3*(1+n)**3*(2+n)) + ((2+n+n*n)*(5-2*S2(n) + 2*(S1(n)*S1(n))))/(n*(1+n)*(2+n))) - 8*CA*nf*TF*((8*(3+2*n)*S1(n))/((1+n)**2*(2+n)**2) + (2*(16+64*n+128*n**3+85*n**4+36*n**5+25*n**6 + 15*n**7+6*n**8+n**9+104*(n*n)))/( (-1+n)*n**3*(1+n)**3*(2+n)**3)+( (2+n+n*n)*(2*S2(n)-2*(S1(n)*S1(n))-2*S2(n/2)))/(n*(1+n)*(2+n))))/4)
    
    gq1 = np.ones_like(p) * ((-(32/3)*CF*nf*TF*((1+n)**(-2) + ((-(8/3)+S1(n))*(2+n+n*n))/((-1+n)*n*(1+n))) - 4*(CF*CF)*((-4*S1(n))/(1+n)**2-( -4-12*n+28*n**3+43*n**4 + 30*n**5+12*n**6-n*n)/((-1+n)*n**3*(1+n)**3) + ((2+n+n*n)*(10*S1(n)-2*S2(n)-2*(S1(n)*S1(n))))/((-1+n)*n*(1+n))) - 8*CF*CA*(((1/9)*(144+432*n-1304*n**3-1031*n**4 + 695*n**5+1678*n**6+1400*n**7+621*n**8+109*n**9 - 152*(n*n)))/((-1+n)**2*n**3*(1+n)**3*(2+n)**2) - ((1/3)*S1(n)*(-12-22*n+17*n**4 + 41*(n*n)))/((-1+n)**2*n**2*(1+n))+( (2+n+n*n)*(S2(n) + S1(n)*S1(n)-S2(n/2)))/((-1+n)*n*(1+n))))/4)
    
    gg1 = np.ones_like(p) * ((CF*nf*TF*(8+(16*(-4-4*n-10*n**3+n**4+4*n**5+2*n**6 - 5*(n*n)))/((-1+n)*n**3*(1+n)**3*(2+n))) + CA*nf*TF*(32/3 - (160/9)*S1(n)+( (16/9)*(12+56*n+76*n**3+38*n**4+94*(n*n)))/((-1+n)*n**2*(1+n)**2*(2+n))) + CA*CA*(-64/3+(536/9)*S1(n)+(64*S1(n)*( -2-2*n+8*n**3+5*n**4+2*n**5+7*(n*n)))/((-1+n)**2*n**2*(1+n)**2*(2+n)**2) - ((4/9)*(576+1488*n-1632*n**3-2344*n**4+1567*n**5 + 6098*n**6+6040*n**7+2742*n**8+457*n**9+560*(n*n)))/( (-1+n)**2*n**3*(1+n)**3*(2+n)**3) - 16*S1(n)*S2(n/2)+(32*(1+n+n*n)*S2(n/2))/( (-1+n)*n*(1+n)*(2+n))-4*S3(n/2) + 32*(S1(n)/n**2-(5/8)*zeta(3)+MellinF2(n) - zeta(2)*(-psi(n/2)+psi((1+n)/2))/2)))/4)
                          
    #qq1_qg1 = np.stack((qq1, qg1), axis=0)
    #gq1_gg1 = np.stack((gq1, gg1), axis=0)
    
    n_tester = np.array([1.0])
    if type(n)==type(n_tester):
        qq1_qg1 = np.stack((qq1, qg1), axis=-1)
        gq1_gg1 = np.stack((gq1, gg1), axis=-1)
        result = np.stack((qq1_qg1,gq1_gg1),axis=-1)
    else:
        length = 1
        result = np.reshape(np.array([[qq1, qg1], [gq1, gg1]]),(length,2,2))

    return result# (N, 2, 2)    

    #return np.array([[qq1, qg1],
    #                 [gq1, gg1]])[np.newaxis,...]*np.ones_like(p)

"""
***********************Evolution operator of GPD in the moment space*******
Refer to the evolution.py at https://github.com/kkumer/gepard. Modifications are made.
"""

def lambdaf(n: complex, nf: int, p: int, prty: int = 1) -> np.ndarray:
    """Eigenvalues of the LO singlet anomalous dimensions matrix.

    Args:
        n (complex): which moment (= Mellin moment for integer n)
        nf (int): number of active quark flavors
        p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et)
        prty (int): 1 for NS^{+}, -1 for NS^{-}, irrelevant at LO

    Returns:
        lam[a, k]
        a in [+, -] and k is MB contour point index

    Normally, n and nf should be scalars. p should be (N)
    More generally, as long as they can be broadcasted, any shape is OK.

    """
    # To avoid crossing of the square root cut on the
    # negative real axis we use trick by Dieter Mueller
    gam0 = singlet_LO(n, nf, p, prty) # (N, 2, 2)
    aux = ((gam0[..., 0, 0] - gam0[..., 1, 1]) *
           np.sqrt(1. + 4.0 * gam0[..., 0, 1] * gam0[..., 1, 0] /
                   (gam0[..., 0, 0] - gam0[..., 1, 1])**2)) # (N)
    lam1 = 0.5 * (gam0[..., 0, 0] + gam0[..., 1, 1] - aux) # (N)
    lam2 = lam1 + aux  # (N)
    return np.stack([lam1, lam2], axis=-1) # shape (N, 2)

def projectors(n: complex, nf: int, p: int, prty: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Projectors on evolution quark-gluon singlet eigenaxes.

    Args:
        n (complex): which moment (= Mellin moment for integer n)
        nf (int): number of active quark flavors
        p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et)
        prty (int): 1 for NS^{+}, -1 for NS^{-}, irrelevant at LO

    Returns:
         lam: eigenvalues of LO an. dimm matrix lam[a, k]  # Eq. (123)
          pr: Projector pr[k, a, i, j]  # Eq. (122)
               k is MB contour point index
               a in [+, -]
               i,j in {Q, G}

    n and nf will be scalars
    p will be shape (N)
    prty should be scalar (but maybe I can make it work with shape N)

    """
    gam0 = singlet_LO(n, nf, p, prty)    # (N, 2, 2)
    lam = lambdaf(n, nf, p, prty)        # (N, 2)
    den = 1. / (lam[..., 0] - lam[..., 1]) #(N)
    # P+ and P-
    ssm = gam0 - np.einsum('...,ij->...ij', lam[..., 1], np.identity(2)) #(N, 2, 2)
    ssp = gam0 - np.einsum('...,ij->...ij', lam[..., 0], np.identity(2)) #(N, 2, 2)
    prp = np.einsum('...,...ij->...ij', den, ssm) # (N, 2, 2)
    prm = np.einsum('...,...ij->...ij', -den, ssp) # (N, 2, 2)
    # We insert a-axis before i,j-axes, i.e. on -3rd place
    pr = np.stack([prp, prm], axis=-3) # (N, 2, 2, 2)
    return lam, pr # (N, 2) and (N, 2, 2, 2)

 
def outer_subtract(arr1,arr2):   
    """Perform the outer product of two array at the last dimension, each has shape (N,..., m)
    
    | Generate shape (N,m,m), Here m = 2 for S/G 
    | result(i,j)=arr1(i)-arr2(j)

    Args:
        arr1 (np.array): 1st array in the outer subtract has shape (N,m)
        arr2 (np.array): 2nd array in the outer subtract has shape (N,m)

    Returns:
        result (np.ndarray): shape(N,m,m) given by result(i,j)=arr1(i)-arr2(j)
    """
    repeated_arr1 = np.repeat(arr1[..., np.newaxis], repeats=2, axis=-1)
    repeated_arr2 = np.repeat(arr2[..., np.newaxis], repeats=2, axis=-1)
    transposed_axes = list(range(repeated_arr1.ndim))
    transposed_axes[-2], transposed_axes[-1] = transposed_axes[-1], transposed_axes[-2]    
    return repeated_arr1-np.transpose(repeated_arr2, axes=transposed_axes)

def rmudep(nf, lamj, lamk, mu):
    """Scale dependent part of NLO evolution matrix 
    
    | Ref to the eq. (126) in hep-ph/0703179 
    | Here the expression is exactly the same as the ref, UNLIKE the Gepard with has an extra beta_0 to be canceled with amuindep

    Args:
        nf (int): number of effective fermions
        lamj (np.array): shape (N,2,2), each row is 2-by-2 matrix of anomalous dimension in the (S, G) basis
        lamk (np.array): shape (N,2,2), second row anomalous dimension for k
        mu (float): final scale to be evolved from inital scale Init_Scale_Q

    Returns:
        R_ij^ab(Q}|n=1) according to eq. (126) in hep-ph/0703179 
    """

    lamdif=outer_subtract(lamj,lamk)
        
    b0 = beta0(nf) # scalar
    b11 = b0 * np.ones_like(lamdif) + lamdif # shape (N,2,2)
    #print(b11)
    R = AlphaS(nloop_alphaS, nf, mu)/AlphaS(nloop_alphaS, nf, Init_Scale_Q) # shape N
    R = np.array(R)
    #print(R)
    Rpow = (1/R)[..., np.newaxis, np.newaxis, np.newaxis]**(b11/b0) # shape (N,2,2)
    #print((np.ones_like(Rpow) - Rpow) / b11)
    return (np.ones_like(Rpow) - Rpow) / b11 # shape (N,2,2)

def amuindep(j: complex, nf: int, p: int, prty: int = 1):
    """Result the P [gamma] P part of the diagonal evolution operator A.
    
    | Ref to eq. (124) in hep-ph/0703179 (the A operator are the same in both CSbar and MSbar scheme)
    | Here the expression is exactly the same as the ref, UNLIKE the Gepard with has an extra 1/beta_0 to be canceled with rmudep
    
    Args:
        j (complex): _description_
        nf (int): _description_
        p (int): _description_
        prty (int, optional): _description_. Defaults to 1.

    Returns:
        the P [gamma] P part of the diagonal evolution operator A.
    """
    lam, pr = projectors(j+1, nf, p, prty)
    
    gam0 = singlet_LO(j+1,nf,p, prty)
    gam1 = singlet_NLO(j+1,nf,p, prty)
    a1 = - gam1 + 0.5 * beta1(nf) * gam0 / beta0(nf)
    A = np.einsum('...aic,...cd,...bdj->...abij', pr, a1, pr)
   
    return A


def evolop(j: complex, nf: int, p: int, mu: float):
    """Leading order GPD evolution operator E(j, nf, mu)[a,b].

    Args:
         j: MB contour points (Note: n = j + 1 !!)
         nf: number of effective fermion
         p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et)
         mu: final scale of evolution 

    Returns:
         Evolution operator E(j, nf, mu)[a,b] at given j nf and mu as 3-by-3 matrix
         - a and b are in the flavor space (non-singlet, singlet, gluon)

    In original evolop function, j, nf, p, and mu are all scalars.
    Here, j and nf will be scalars.
    p and mu will have shape (N)

    """
    #Alpha-strong ratio.
    R = AlphaS(nloop_alphaS, nf, mu)/AlphaS(nloop_alphaS, nf, Init_Scale_Q) # shape N
    R = np.array(R)

    #LO singlet anomalous dimensions and projectors
    lam, pr = projectors(j+1, nf, p)    # (N, 2) (N, 2, 2, 2)

    #LO pQCD beta function of GPD evolution
    b0 = beta0(nf) # scalar

    #Singlet LO evolution factor (alpha(mu)/alpha(mu0))^(-gamma/beta0) in (+,-) space
    Rfact = R[..., np.newaxis]**(-lam/b0) # (N, 2)     

    #Singlet LO evolution matrix in (u+d, g) space
    """
    # The Gepard code by K. Kumericki reads:
    evola0ab = np.einsum('kaij,ab->kabij', pr,  np.identity(2))
    evola0 = np.einsum('kabij,bk->kij', evola0ab, Rfact)
    # We use instead
    """ 
    evola0 = np.einsum('...aij,...a->...ij', pr, Rfact) # (N, 2, 2)
    
    #Non-singlet LO anomalous dimension
    gam0NS = non_singlet_LO(j+1, nf, p) #this function is already numpy compatible
    # shape (N)

    #Non-singlet evolution factor 
    evola0NS = R**(-gam0NS/b0) #(N)

    return [evola0NS, evola0] # (N) and (N, 2, 2)

# Need Wilson coefficients for evolution. Allows numerical pre-calculation of non-diagonal piece using Mellin-Barnes integral

def WilsonCoef(j: complex) -> complex:
    """Leading-order Wilson coefficient, elements used in both DVCS and DVMP

    Args:
        j (complex array): shape(N,) conformal spin j

    Returns:
        Leading-order Wilson coefficient (complex, could be an array)
    """
    return 2 ** (1+j) * gamma(5/2+j) / (gamma(3/2) * gamma(3+j))

def WilsonCoef_DVCS_LO(j: complex) -> complex:
    """LO Wilson coefficient of DVCS in the evolution basis (qVal, q_du_plus, q_du_minus, qSigma, g)
        
    Args:
        j (complex array): shape(N,) conformal spin j
        
    Returns:
        Wilson coefficient of shape (5,N) in the evolution basis (qVal, q_du_plus, q_du_minus, qSigma, g)
        
    | Charge factor are calculated such that the sum in the evolution basis are identical to the sum in the flavor basis
    | Gluon charge factor is the same as the singlet one, but the LO Wilson coefficient is zero in DVCS.
    """
    charge_fact = np.array([0, -1/6, 0, 5/18, 5/18])
    CWT = np.array([WilsonCoef(j), \
                    WilsonCoef(j), \
                    WilsonCoef(j), \
                    WilsonCoef(j),\
                    0 * j])
    return np.einsum('j, j...->j...', charge_fact, CWT)

def WilsonCoef_DVCS_NLO(j: complex, nf: int, Q: float, mu: float, p:int) -> complex:
    """NLO Wilson coefficient of DVCS in the evolution basis (qVal, q_du_plus, q_du_minus, qSigma, g)

    Check eqs. (127)-(130) of https://arxiv.org/pdf/hep-ph/0703179 
    
    Args:
        j (complex array): shape(N,) conformal spin j
        nf (int): number of effective fermions
        Q (float): the photon virtuality 
        mu (float): the factorization scale mu_fact
        p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et), scalars
        
    Returns:
        Wilson coefficient of shape (5,N) in the evolution basis (qVal, q_du_plus, q_du_minus, qSigma, g)
        
    | Charge factor are calculated such that the sum in the evolution basis are identical to the sum in the flavor basis
    | Gluon charge factor is the same as the singlet one, but the LO Wilson coefficient is zero in DVCS.
    """
    charge_fact = np.array([0, -1/6, 0, 5/18, 5/18])
    gam0 = singlet_LO(j+1, nf, p)
    qq0 = gam0[...,0,0]
    qg0 = gam0[...,0,1]
    
    if(p == 1):
        CWT = np.array([WilsonCoef(j) * (CF * (2 * (S1(1+j)) ** 2 - 9/2 + (5 - 4* S1(j+1))/2/(j+1)/(j+2) + 1/(j+1)**2/(j+2)**2)+ np.log(mu**2/Q**2)/2*qq0), \
                        WilsonCoef(j) * (CF * (2 * (S1(1+j)) ** 2 - 9/2 + (5 - 4* S1(j+1))/2/(j+1)/(j+2) + 1/(j+1)**2/(j+2)**2)+ np.log(mu**2/Q**2)/2*qq0), \
                        WilsonCoef(j) * (CF * (2 * (S1(1+j)) ** 2 - 9/2 + (5 - 4* S1(j+1))/2/(j+1)/(j+2) + 1/(j+1)**2/(j+2)**2)+ np.log(mu**2/Q**2)/2*qq0), \
                        WilsonCoef(j) * (CF * (2 * (S1(1+j)) ** 2 - 9/2 + (5 - 4* S1(j+1))/2/(j+1)/(j+2) + 1/(j+1)**2/(j+2)**2)+ np.log(mu**2/Q**2)/2*qq0),\
                        WilsonCoef(j) * (-nf *((4 + 3*j + j**2) * (S1(j)+S1(j+2)) + 2+3*j+j**2)/(j+1)/(j+2)/(j+3) + np.log(mu**2/Q**2)/2*qg0)])

    else:
        CWT = np.array([WilsonCoef(j) * (CF * (2 * (S1(1+j)) ** 2 - 9/2 + (3 - 4* S1(j+1))/2/(j+1)/(j+2) + 1/(j+1)**2/(j+2)**2)+ np.log(mu**2/Q**2)/2*qq0), \
                        WilsonCoef(j) * (CF * (2 * (S1(1+j)) ** 2 - 9/2 + (3 - 4* S1(j+1))/2/(j+1)/(j+2) + 1/(j+1)**2/(j+2)**2)+ np.log(mu**2/Q**2)/2*qq0), \
                        WilsonCoef(j) * (CF * (2 * (S1(1+j)) ** 2 - 9/2 + (3 - 4* S1(j+1))/2/(j+1)/(j+2) + 1/(j+1)**2/(j+2)**2)+ np.log(mu**2/Q**2)/2*qq0), \
                        WilsonCoef(j) * (CF * (2 * (S1(1+j)) ** 2 - 9/2 + (3 - 4* S1(j+1))/2/(j+1)/(j+2) + 1/(j+1)**2/(j+2)**2)+ np.log(mu**2/Q**2)/2*qq0),\
                        WilsonCoef(j) * (-nf * j * (1 + S1(j) + S1(j+2))/(j+1)/(j+2) + np.log(mu**2/Q**2)/2*qg0)])
        
    return np.einsum('j, j...->j...', charge_fact, CWT)


def Evo_SG_NLO(j: np.array, nf: int, p: int, mu: float) -> np.array:
    """FORWARD Next-to-leading order evolved conformal moments in the evolution basis (Evolved moment method)    
    
    This function removes the off-diagonal pieces in Moment_Evo_NLO()
    
    Args:
        j: conformal spin j (conformal spin is actually j+2 but anyway): array (N,)
        nf: number of effective fermions; 
        p (int): 1 for vector-like GPD (Ht, Et), -1 for axial-vector-like GPDs (Ht, Et): scalar
        mu: final evolution scale: scalar
        ConfFlav: unevolved conformal moments

    Returns:
        Next-to-leading order evolved conformal moments in the evolution basis in the forward limit (to be combined with inverse Mellin transform wave function)
        return shape (N, 5)
        
    | The j here accept array input and preferrably just 1D for the contour integral in j. 
    | Therefore, j has shape (N,) where N is the interpolating order of the fixed quad.
    | Other quantities must be broadcastable with j and thus they should be preferrably scalar
    """
    
    assert j.ndim == 1, "Check dimension of j, must be 1D array" # shape (N,)
   
    # Set up evolution operator for WCs
    Alphafact = np.array(AlphaS(nloop_alphaS, nf, mu)) / np.pi / 2
    R = AlphaS(nloop_alphaS, nf, mu)/AlphaS(nloop_alphaS, nf, Init_Scale_Q) # shape N
    R = np.array(R)
  
    b0 = beta0(nf)
    lam, pr = projectors(j+1, nf, p)
    pproj = amuindep(j, nf, p)
    
    rmu1 = rmudep(nf, lam, lam, mu)

    Rfact = R[...,np.newaxis]**(-lam/b0)  # LO evolution (alpha(mu)/alpha(mu0))^(-gamma/beta0)

    # S/G LO evolution operator
    evola0 = np.einsum('...aij,...a->...ij', pr, Rfact)
    
    # S/G diagonal NLO evolution operator    
    evola1_diag_ab = np.einsum('kab,kabij->kabij', rmu1, pproj)
    evola1_diag = np.einsum('...abij,...b,...->...ij', evola1_diag_ab, Rfact,Alphafact)
    
    return evola0, evola1_diag

def Evo_WilsonCoef_SG(mu: float,nf: int, p:int = 1, p_order: int =1):
    """Return evolved Wilson Coefficient at j=1

    Args:
        mu (float): scale evolved to, initial scale=M_jpsi/2
        nf (int): number of effective fermion =4 for u d s c
        p (int): p=1 for vector-like
        p_order (int): 1 for LO, 2 for NLO, 3 for partially NNLO (including the C^{NLO} E^{NLO} term)

    Returns:
        (C_Sigma, C_G) at j=0: Evolve Wilson Coefficient 
    """
    j0 = np.array([1.])    
    alphaS = AlphaS(nloop_alphaS, nf, mu)        
    
    m_charm = M_jpsi/2      
    cg1 = -0.369
    cq1 = 0.891
    
    evola0, evola1_diag = Evo_SG_NLO(j0,nf,p,mu)
    
    evola0= evola0[0]
    evola1_diag= evola1_diag[0]
    
    CWj0qLO = 0 
    CWj0gLO = 5/4 
    CWj0LO = [CWj0qLO, CWj0gLO]
    
    CWj0qNLO = alphaS * (cq1 - 10/9/np.pi* 2 * log(m_charm/mu))
    CWj0gNLO = alphaS * (cg1 - 55/16/np.pi * 2 * log(m_charm/mu))
    CWj0NLO = [CWj0qNLO, CWj0gNLO]
    
    CWevo0 = alphaS * np.einsum('ij,j->i',evola0,CWj0LO)
    
    CWevo1 = alphaS * np.einsum('ij,j->i',evola1_diag,CWj0LO)\
            + alphaS * np.einsum('ij,j->i',evola0,CWj0NLO)
    
    CWevo2 =  alphaS * np.einsum('ij,j->i',evola1_diag,CWj0NLO)
    
    if(p_order == 1):
        return CWevo0
    if(p_order == 2):
        return CWevo0 + CWevo1
    if(p_order == 3):
        return CWevo0 + CWevo1 + CWevo2

