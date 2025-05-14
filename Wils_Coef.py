from numpy import log as Log
from numpy import sqrt as Sqrt
from numpy import pi as Pi
from mpmath import polylog as PolyLog
from mpmath import hyp2f1
from scipy.special import gamma
import numpy as np
from scipy.integrate import quad

hyp2f1_nparray = np.frompyfunc(hyp2f1,4,1)


def ConfWaveFuncQ(j: complex, x: float, xi: float) -> complex:
    """Quark conformal wave function p_j(x,xi) 
    
    Check e.g. https://arxiv.org/pdf/hep-ph/0509204.pdf

    Args:
        j: conformal spin j (conformal spin is actually j+2 but anyway)
        x: momentum fraction x
        xi: skewness parameter xi

    Returns:
        quark conformal wave function p_j(x,xi)
    """  

    pDGLAP = np.where(x >= xi,                np.sin(np.pi * (j+1))/ np.pi * x**(-j-1) * np.array(hyp2f1_nparray( (j+1)/2, (j+2)/2, j+5/2, (xi/x) ** 2), dtype= complex)                           , 0)

    pERBL = np.where(((x > -xi) & (x < xi)), 2 ** (1+j) * gamma(5/2+j) / (gamma(1/2) * gamma(1+j)) * xi ** (-j-1) * (1+x/xi) * np.array(hyp2f1_nparray(-1-j,j+2,2, (x+xi)/(2*xi)), dtype= complex), 0)

    return pDGLAP + pERBL

def ConfWaveFuncG(j: complex, x: float, xi: float) -> complex:
    """Gluon conformal wave function p_j(x,xi) 
    
    Check e.g. https://arxiv.org/pdf/hep-ph/0509204.pdf

    Args:
        j: conformal spin j (actually conformal spin is j+2 but anyway)
        x: momentum fraction x
        xi: skewness parameter xi

    Returns:
        gluon conformal wave function p_j(x,xi)
    """ 
    # An extra minus sign defined different from the orginal definition to absorb the extra minus sign of MB integral for gluon
    
    Minus = -1

    pDGLAP = np.where(x >= xi,                Minus * np.sin(np.pi * j)/ np.pi * x**(-j) * np.array(hyp2f1_nparray( j/2, (j+1)/2, j+5/2, (xi/x) ** 2), dtype= complex)                                   , 0)

    pERBL = np.where(((x > -xi) & (x < xi)), Minus * 2 ** j * gamma(5/2+j) / (gamma(1/2) * gamma(j)) * xi ** (-j) * (1+x/xi) ** 2 * np.array((hyp2f1_nparray(-1-j,j+2,3, (x+xi)/(2*xi))), dtype= complex), 0)

    return pDGLAP + pERBL

"""

Note: The following expression in this file are NOT meant to be readable!

Convert from the Mathematica code of https://arxiv.org/abs/2105.07657 in the Q->0 limit

Numerically check with the Mathematica version

"""

def C1Tg(x: float, xi: float, delta: float):
    """ Leading gluon wilson coefficient for transversely polarized photon, Check eq. (2.19) for definition
    
        1/x**2 is absorbed into the defition, such that the leading Wilson coefficient is normalized as 1/(xi^2-x^2)

    Args:
        x (float): parton momentum fraction x
        xi (float): skewness xi
        delta (float): IR regulator xi -> xi-i delta

    Returns:
        C1Tg (complex): Leading gluon wilson coefficient for transversely polarized photon
    """
    
    return 1/(x**2 + (delta + complex(0,1)*xi)**2)

def C2Tgcon(x: float, xi: float, delta: float):
    """ Next-to-leading gluon wilson coefficient for transversely polarized photon, check eq. (2.19) for definition
    
        | 1/x**2 is absorbed into the defition, such that the leading Wilson coefficient is normalized as 1/(xi^2-x^2)
        | EXCLUDING the log(m^2/muf^2) terms which will be implemented separately in the following.

    Args:
        x (float): parton momentum fraction x
        xi (float): skewness xi
        delta (float): IR regulator xi -> xi-i delta

    Returns:
        C2Tgcon (complex): Next-to-leading gluon wilson coefficient for transversely polarized photon excluding the log(m^2/muf^2) terms
    """
    
    return (768*x**4*(complex(0,1)*delta - xi)**3 + 109*Pi**2*x**4*(complex(0,1)*delta - xi)**3 + 384*x**6*(complex(0,-1)*delta + xi) + 101*Pi**2*x**6*(complex(0,-1)*delta + xi) + 384*x**2*(complex(0,-1)*delta + xi)**5 + complex(0,936)*x**4*(delta + complex(0,1)*xi)**3*Log(2) + complex(0,1320)*x**2*(delta + complex(0,1)*xi)**5*Log(2) + 384*x**6*(complex(0,-1)*delta + xi)*Log(2) + 600*x**4*(complex(0,1)*delta - xi)**3*Log(2)**2 + complex(0,432)*x**6*(delta + complex(0,1)*xi)*Log(2)**2 - 168*x**5*(delta + complex(0,1)*xi)**2*Log(2)**2 - 312*x**3*(delta + complex(0,1)*xi)**4*Log(2)**2 + 888*x**2*(complex(0,-1)*delta + xi)**5*Log(2)**2 + 396*x**4*(complex(0,1)*delta - xi)**3*Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi))*Log(4) - 324*x**5*(delta + complex(0,1)*xi)**2*Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi))*Log(4) - 324*x**3*(delta + complex(0,1)*xi)**4*Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi))*Log(4) + 396*x**2*(complex(0,-1)*delta + xi)**5*Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi))*Log(4) + 816*x**5*(delta + complex(0,1)*xi)**2*Log((complex(0,1)*delta - xi)/x) + complex(0,528)*x**4*(delta + complex(0,1)*xi)**3*Log((complex(0,1)*delta - xi)/x) + 936*x**3*(delta + complex(0,1)*xi)**4*Log((complex(0,1)*delta - xi)/x) + complex(0,684)*x**2*(delta + complex(0,1)*xi)**5*Log((complex(0,1)*delta - xi)/x) + 120*x*(delta + complex(0,1)*xi)**6*Log((complex(0,1)*delta - xi)/x) + complex(0,24)*(delta + complex(0,1)*xi)**7*Log((complex(0,1)*delta - xi)/x) + 132*x**6*(complex(0,-1)*delta + xi)*Log((complex(0,1)*delta - xi)/x) - 216*x**7*Log((complex(0,1)*delta - xi)/x)**2 + 216*x**3*(delta + complex(0,1)*xi)**4*Log((complex(0,1)*delta - xi)/x)**2 + complex(0,216)*x**2*(delta + complex(0,1)*xi)**5*Log((complex(0,1)*delta - xi)/x)**2 + 216*x**6*(complex(0,-1)*delta + xi)*Log((complex(0,1)*delta - xi)/x)**2 + 528*x**4*(complex(0,1)*delta - xi)**3*Log((complex(0,1)*delta + x - xi)/x) + 24*(complex(0,1)*delta - xi)**7*Log((complex(0,1)*delta + x - xi)/x) + complex(0,132)*x**6*(delta + complex(0,1)*xi)*Log((complex(0,1)*delta + x - xi)/x) - 816*x**5*(delta + complex(0,1)*xi)**2*Log((complex(0,1)*delta + x - xi)/x) - 936*x**3*(delta + complex(0,1)*xi)**4*Log((complex(0,1)*delta + x - xi)/x) - 120*x*(delta + complex(0,1)*xi)**6*Log((complex(0,1)*delta + x - xi)/x) + 684*x**2*(complex(0,-1)*delta + xi)**5*Log((complex(0,1)*delta + x - xi)/x) + 432*x**7*Log((complex(0,1)*delta - xi)/x)*Log((complex(0,1)*delta + x - xi)/x) + complex(0,432)*x**6*(delta + complex(0,1)*xi)*Log((complex(0,1)*delta - xi)/x)*Log((complex(0,1)*delta + x - xi)/x) - 432*x**3*(delta + complex(0,1)*xi)**4*Log((complex(0,1)*delta - xi)/x)*Log((complex(0,1)*delta + x - xi)/x) + 432*x**2*(complex(0,-1)*delta + xi)**5*Log((complex(0,1)*delta - xi)/x)*Log((complex(0,1)*delta + x - xi)/x) - 216*x**7*Log((complex(0,1)*delta + x - xi)/x)**2 + 216*x**3*(delta + complex(0,1)*xi)**4*Log((complex(0,1)*delta + x - xi)/x)**2 + complex(0,216)*x**2*(delta + complex(0,1)*xi)**5*Log((complex(0,1)*delta + x - xi)/x)**2 + 216*x**6*(complex(0,-1)*delta + xi)*Log((complex(0,1)*delta + x - xi)/x)**2 - 816*x**5*(delta + complex(0,1)*xi)**2*Log((complex(0,-1)*delta + xi)/x) + complex(0,528)*x**4*(delta + complex(0,1)*xi)**3*Log((complex(0,-1)*delta + xi)/x) - 936*x**3*(delta + complex(0,1)*xi)**4*Log((complex(0,-1)*delta + xi)/x) + complex(0,684)*x**2*(delta + complex(0,1)*xi)**5*Log((complex(0,-1)*delta + xi)/x) - 120*x*(delta + complex(0,1)*xi)**6*Log((complex(0,-1)*delta + xi)/x) + complex(0,24)*(delta + complex(0,1)*xi)**7*Log((complex(0,-1)*delta + xi)/x) + 132*x**6*(complex(0,-1)*delta + xi)*Log((complex(0,-1)*delta + xi)/x) + 216*x**7*Log((complex(0,-1)*delta + xi)/x)**2 - 216*x**3*(delta + complex(0,1)*xi)**4*Log((complex(0,-1)*delta + xi)/x)**2 + complex(0,216)*x**2*(delta + complex(0,1)*xi)**5*Log((complex(0,-1)*delta + xi)/x)**2 + 216*x**6*(complex(0,-1)*delta + xi)*Log((complex(0,-1)*delta + xi)/x)**2 + 528*x**4*(complex(0,1)*delta - xi)**3*Log((complex(0,-1)*delta + x + xi)/x) + 24*(complex(0,1)*delta - xi)**7*Log((complex(0,-1)*delta + x + xi)/x) + complex(0,132)*x**6*(delta + complex(0,1)*xi)*Log((complex(0,-1)*delta + x + xi)/x) + 816*x**5*(delta + complex(0,1)*xi)**2*Log((complex(0,-1)*delta + x + xi)/x) + 936*x**3*(delta + complex(0,1)*xi)**4*Log((complex(0,-1)*delta + x + xi)/x) + 120*x*(delta + complex(0,1)*xi)**6*Log((complex(0,-1)*delta + x + xi)/x) + 684*x**2*(complex(0,-1)*delta + xi)**5*Log((complex(0,-1)*delta + x + xi)/x) - 432*x**7*Log((complex(0,-1)*delta + xi)/x)*Log((complex(0,-1)*delta + x + xi)/x) + complex(0,432)*x**6*(delta + complex(0,1)*xi)*Log((complex(0,-1)*delta + xi)/x)*Log((complex(0,-1)*delta + x + xi)/x) + 432*x**3*(delta + complex(0,1)*xi)**4*Log((complex(0,-1)*delta + xi)/x)*Log((complex(0,-1)*delta + x + xi)/x) + 432*x**2*(complex(0,-1)*delta + xi)**5*Log((complex(0,-1)*delta + xi)/x)*Log((complex(0,-1)*delta + x + xi)/x) + 216*x**7*Log((complex(0,-1)*delta + x + xi)/x)**2 - 216*x**3*(delta + complex(0,1)*xi)**4*Log((complex(0,-1)*delta + x + xi)/x)**2 + complex(0,216)*x**2*(delta + complex(0,1)*xi)**5*Log((complex(0,-1)*delta + x + xi)/x)**2 + 216*x**6*(complex(0,-1)*delta + xi)*Log((complex(0,-1)*delta + x + xi)/x)**2 - 324*x**5*(delta + complex(0,1)*xi)**2*Sqrt((complex(0,1)*delta + x - xi)/(complex(0,-1)*delta + x + xi))*Log(((complex(0,-1)*delta + x + xi)*(x/(complex(0,-1)*delta + x + xi) + Sqrt(-1 + (2*x)/(complex(0,-1)*delta + x + xi))))/(complex(0,-1)*delta + xi)) + complex(0,396)*x**4*(delta + complex(0,1)*xi)**3*Sqrt((complex(0,1)*delta + x - xi)/(complex(0,-1)*delta + x + xi))*Log(((complex(0,-1)*delta + x + xi)*(x/(complex(0,-1)*delta + x + xi) + Sqrt(-1 + (2*x)/(complex(0,-1)*delta + x + xi))))/(complex(0,-1)*delta + xi)) - 324*x**3*(delta + complex(0,1)*xi)**4*Sqrt((complex(0,1)*delta + x - xi)/(complex(0,-1)*delta + x + xi))*Log(((complex(0,-1)*delta + x + xi)*(x/(complex(0,-1)*delta + x + xi) + Sqrt(-1 + (2*x)/(complex(0,-1)*delta + x + xi))))/(complex(0,-1)*delta + xi)) + complex(0,396)*x**2*(delta + complex(0,1)*xi)**5*Sqrt((complex(0,1)*delta + x - xi)/(complex(0,-1)*delta + x + xi))*Log(((complex(0,-1)*delta + x + xi)*(x/(complex(0,-1)*delta + x + xi) + Sqrt(-1 + (2*x)/(complex(0,-1)*delta + x + xi))))/(complex(0,-1)*delta + xi)) + 150*x**4*(complex(0,1)*delta - xi)**3*Log(((complex(0,-1)*delta + x + xi)*(x/(complex(0,-1)*delta + x + xi) + Sqrt(-1 + (2*x)/(complex(0,-1)*delta + x + xi))))/(complex(0,-1)*delta + xi))**2 + 42*x**5*(delta + complex(0,1)*xi)**2*Log(((complex(0,-1)*delta + x + xi)*(x/(complex(0,-1)*delta + x + xi) + Sqrt(-1 + (2*x)/(complex(0,-1)*delta + x + xi))))/(complex(0,-1)*delta + xi))**2 + 78*x**3*(delta + complex(0,1)*xi)**4*Log(((complex(0,-1)*delta + x + xi)*(x/(complex(0,-1)*delta + x + xi) + Sqrt(-1 + (2*x)/(complex(0,-1)*delta + x + xi))))/(complex(0,-1)*delta + xi))**2 + 114*x**2*(complex(0,-1)*delta + xi)**5*Log(((complex(0,-1)*delta + x + xi)*(x/(complex(0,-1)*delta + x + xi) + Sqrt(-1 + (2*x)/(complex(0,-1)*delta + x + xi))))/(complex(0,-1)*delta + xi))**2 + 324*x**5*(delta + complex(0,1)*xi)**2*Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi))*Log(4*Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi)) - (complex(0,4)*x*(1 + Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi))))/(delta + complex(0,1)*xi)) + complex(0,396)*x**4*(delta + complex(0,1)*xi)**3*Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi))*Log(4*Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi)) - (complex(0,4)*x*(1 + Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi))))/(delta + complex(0,1)*xi)) + 324*x**3*(delta + complex(0,1)*xi)**4*Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi))*Log(4*Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi)) - (complex(0,4)*x*(1 + Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi))))/(delta + complex(0,1)*xi)) + complex(0,396)*x**2*(delta + complex(0,1)*xi)**5*Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi))*Log(4*Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi)) - (complex(0,4)*x*(1 + Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi))))/(delta + complex(0,1)*xi)) + 168*x**5*(delta + complex(0,1)*xi)**2*Log(2)*Log(4*Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi)) - (complex(0,4)*x*(1 + Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi))))/(delta + complex(0,1)*xi)) + complex(0,600)*x**4*(delta + complex(0,1)*xi)**3*Log(2)*Log(4*Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi)) - (complex(0,4)*x*(1 + Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi))))/(delta + complex(0,1)*xi)) + 312*x**3*(delta + complex(0,1)*xi)**4*Log(2)*Log(4*Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi)) - (complex(0,4)*x*(1 + Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi))))/(delta + complex(0,1)*xi)) + complex(0,456)*x**2*(delta + complex(0,1)*xi)**5*Log(2)*Log(4*Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi)) - (complex(0,4)*x*(1 + Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi))))/(delta + complex(0,1)*xi)) + 150*x**4*(complex(0,1)*delta - xi)**3*Log(4*Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi)) - (complex(0,4)*x*(1 + Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi))))/(delta + complex(0,1)*xi))**2 - 42*x**5*(delta + complex(0,1)*xi)**2*Log(4*Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi)) - (complex(0,4)*x*(1 + Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi))))/(delta + complex(0,1)*xi))**2 - 78*x**3*(delta + complex(0,1)*xi)**4*Log(4*Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi)) - (complex(0,4)*x*(1 + Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi))))/(delta + complex(0,1)*xi))**2 + 114*x**2*(complex(0,-1)*delta + xi)**5*Log(4*Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi)) - (complex(0,4)*x*(1 + Sqrt((complex(0,-1)*delta + x + xi)/(complex(0,1)*delta + x - xi))))/(delta + complex(0,1)*xi))**2 + 24*x*(18*x**6 + 3*x**4*(delta + complex(0,1)*xi)**2 - complex(0,7)*x**3*(delta + complex(0,1)*xi)**3 + 2*(delta + complex(0,1)*xi)**6 + 11*x**5*(complex(0,-1)*delta + xi) + 3*x*(complex(0,-1)*delta + xi)**5)*PolyLog(2,-(x/(complex(0,-1)*delta + xi))) - 24*x*(18*x**6 + complex(0,11)*x**5*(delta + complex(0,1)*xi) + 3*x**4*(delta + complex(0,1)*xi)**2 + complex(0,7)*x**3*(delta + complex(0,1)*xi)**3 + complex(0,3)*x*(delta + complex(0,1)*xi)**5 + 2*(delta + complex(0,1)*xi)**6)*PolyLog(2,x/(complex(0,-1)*delta + xi)))/(144.*Pi*x**8*(complex(0,-1)*delta + xi)*(-1 + (complex(0,-1)*delta + xi)**2/x**2)**3)

def C2Tglog(x: float, xi: float, delta: float):
    """Next-to-leading gluon wilson coefficient for transversely polarized photon, check eq. (2.19) for definition
    
        | 1/x**2 is absorbed into the defition, such that the leading Wilson coefficient is normalized as 1/(xi^2-x^2)
        | ONLY the log(m^2/muf^2) terms here (log(m^2/muf^2) shoud be multiplied separately)

    Args:
        x (float): parton momentum fraction x
        xi (float): skewness xi
        delta (float): IR regulator xi -> xi-i delta

    Returns:
        C2Tglog(complex): Next-to-leading gluon wilson coefficient for transversely polarized photon with ONLY the log(m^2/muf^2) terms
    """
    
    return (complex(0,-1.5)*(x**2 - (delta + complex(0,1)*xi)**2)*(complex(0,2)*(delta + complex(0,1)*xi)*Log(complex(0,2)*delta - 2*xi) + (complex(0,-1)*delta + x + xi)*Log(complex(0,1)*delta - x - xi) + (complex(0,-1)*delta - x + xi)*Log(complex(0,1)*delta + x - xi)))/(Pi*(x**2 + (delta + complex(0,1)*xi)**2)**2*(delta + complex(0,1)*xi))
    
def C2Tqcon(x: float, xi: float, delta: float):
    """Next-to-leading quark wilson coefficient for transversely polarized photon, check eq. (2.19) for definition
    
        | 1/x is absorbed into the defition.
        | EXCLUDING the log(m^2/muf^2) terms which will be implemented separately in the following.
        
    Args:
        x (float): parton momentum fraction x
        xi (float): skewness xi
        delta (float): IR regulator xi -> xi-i delta

    Returns:
        C2Tqcon(complex): Next-to-leading quark wilson coefficient for transversely polarized photon excluding the log(m^2/muf^2) terms
    """
    
    return -1/18*((13*Pi**2*(complex(0,-1)*delta + xi))/x + (complex(0,48)*(delta + complex(0,1)*xi)*Log(2)**2)/x + (12*(complex(0,-1)*delta + xi)*Log((complex(0,1)*delta - xi)/x))/x - (12*(complex(0,-1)*delta + xi)**3*Log((complex(0,1)*delta - xi)/x))/x**3 - 24*Log((complex(0,1)*delta - xi)/x)**2 + (24*(complex(0,-1)*delta + xi)*Log((complex(0,1)*delta - xi)/x)**2)/x + (complex(0,12)*(delta + complex(0,1)*xi)*Log((complex(0,1)*delta + x - xi)/x))/x + (12*(complex(0,-1)*delta + xi)**3*Log((complex(0,1)*delta + x - xi)/x))/x**3 + 48*Log((complex(0,1)*delta - xi)/x)*Log((complex(0,1)*delta + x - xi)/x) + (complex(0,48)*(delta + complex(0,1)*xi)*Log((complex(0,1)*delta - xi)/x)*Log((complex(0,1)*delta + x - xi)/x))/x - 24*Log((complex(0,1)*delta + x - xi)/x)**2 + (24*(complex(0,-1)*delta + xi)*Log((complex(0,1)*delta + x - xi)/x)**2)/x + (12*(complex(0,-1)*delta + xi)*Log((complex(0,-1)*delta + xi)/x))/x - (12*(complex(0,-1)*delta + xi)**3*Log((complex(0,-1)*delta + xi)/x))/x**3 + 24*Log((complex(0,-1)*delta + xi)/x)**2 + (24*(complex(0,-1)*delta + xi)*Log((complex(0,-1)*delta + xi)/x)**2)/x + (complex(0,12)*(delta + complex(0,1)*xi)*Log((complex(0,-1)*delta + x + xi)/x))/x + (12*(complex(0,-1)*delta + xi)**3*Log((complex(0,-1)*delta + x + xi)/x))/x**3 - 48*Log((complex(0,-1)*delta + xi)/x)*Log((complex(0,-1)*delta + x + xi)/x) + (complex(0,48)*(delta + complex(0,1)*xi)*Log((complex(0,-1)*delta + xi)/x)*Log((complex(0,-1)*delta + x + xi)/x))/x + 24*Log((complex(0,-1)*delta + x + xi)/x)**2 + (24*(complex(0,-1)*delta + xi)*Log((complex(0,-1)*delta + x + xi)/x)**2)/x + 12*(4 + (2*(complex(0,-1)*delta + xi))/x + (complex(0,-1)*delta + xi)**2/x**2)*PolyLog(2,-(x/(complex(0,-1)*delta + xi))) - 12*(4 + (complex(0,2)*delta - 2*xi)/x + (complex(0,-1)*delta + xi)**2/x**2)*PolyLog(2,x/(complex(0,-1)*delta + xi)))/(Pi*(complex(0,-1)*delta + xi)*(-1 + (complex(0,-1)*delta + xi)**2/x**2))

def C2Tqlog(x: float, xi: float, delta: float):
    """ Next-to-leading quark wilson coefficient for transversely polarized photon, check eq. (2.19) for definition
        
        | 1/x is absorbed into the defition.
        | ONLY the log(m^2/muf^2) terms here (log(m^2/muf^2) shoud be multiplied separately)
        
    Args:
        x (float): parton momentum fraction x
        xi (float): skewness xi
        delta (float): IR regulator xi -> xi-i delta

    Returns:
        C2Tqlog(complex): Next-to-leading quark wilson coefficient for transversely polarized photon with ONLY the log(m^2/muf^2) terms
    """
    
    return (4*x*(complex(0,-1)*xi*Log(4) - delta*Log(complex(0,4)*delta - 4*xi) - (delta + complex(0,2)*xi)*Log(complex(0,1)*delta - xi) + (delta + complex(0,1)*x + complex(0,1)*xi)*Log(complex(0,1)*delta - x - xi) + (delta - complex(0,1)*x + complex(0,1)*xi)*Log(complex(0,1)*delta + x - xi)))/(3.*Pi*(x**2 + (delta + complex(0,1)*xi)**2)*(delta + complex(0,1)*xi))


def Conf_ConvertorQ(j: complex, xi: float, delta: float, WilsonCoef):
    """ Convert Wilson coefficient to conformal space

    Args:
        j (complex): conformal spin that could be complex
        xi (float): skewness parameter xi
        delta (float): IR regulator xi -> xi-i deltaescription_
        WilsonCoef (func): C(x,xi,delta) that gives the x-space Wilson coefficient to be converted.
        
    Returns:
        C(j,xi,delta) (complex): Wilson coefficient convert to conformal space
    """
    def integrand(x):
        
        return WilsonCoef(x,xi,delta) * ConfWaveFuncQ(j,x,xi)
    
    def complex_int(func_int, a, b):
        
        real, real_err =  quad(lambda x: np.real(func_int(x)), a, b)
        imag, imag_err =  quad(lambda x: np.imag(func_int(x)), a, b)
        
        return real+1j*imag, real_err+1j*imag_err
    
    result, error = complex_int(integrand,-xi,np.inf)
    
    return result, error

def Conf_ConvertorG(j: complex, xi: float, delta: float, WilsonCoef):
    """ Convert Wilson coefficient to conformal space

    Args:
        j (complex): conformal spin that could be complex
        xi (float): skewness parameter xi
        delta (float): IR regulator xi -> xi-i deltaescription_
        WilsonCoef (func): C(x,xi,delta) that gives the x-space Wilson coefficient to be converted.
        
    Returns:
        C(j,xi,delta) (complex): Wilson coefficient convert to conformal space
    """
    def integrand(x):
        
        return WilsonCoef(x,xi,delta) * ConfWaveFuncG(j,x,xi)
    
    def complex_int(func_int, a, b):
        
        real, real_err =  quad(lambda x: np.real(func_int(x)), a, b)
        imag, imag_err =  quad(lambda x: np.imag(func_int(x)), a, b)
        
        return real+1j*imag, real_err+1j*imag_err
    
    result, error = complex_int(integrand,-xi,np.inf)
    
    return result, error