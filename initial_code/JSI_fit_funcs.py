import numpy as np
import scipy

def JSI_nofil(omega_s, omega_i,omega_0s,omega_0i, sigma_p, A):
    nu_s = omega_s-omega_0s
    nu_i = omega_i-omega_0i
    return A*np.exp(-(nu_s/sigma_p)**2-(nu_i/sigma_p)**2 - 2*nu_s*nu_i/(sigma_p*sigma_p))

def _JSI_nofil(M, *args):
    x, y = M[:,0],M[:,0]
    arr = np.zeros(x.shape)
    arr+=JSI_nofil(x,y,*args)
    return arr

def JSI_fil(omega_s, omega_i,omega_0s,omega_0i,sigma_p,sigma_fs, sigma_fi, A):
    nu_s = omega_s-omega_0s
    nu_i = omega_i-omega_0i
    return JSI_nofil(omega_s, omega_i,omega_0s,omega_0i,sigma_p, A)*np.exp(-(nu_s/sigma_fs)**2)*np.exp(-(nu_i/sigma_fi)**2)

def _JSI_fil(M, *args):
    x, y = M
    return JSI_fil(x,y,*args)

def K(sigma, sigma_f):
    return 1/np.sqrt(1-1/(1+(sigma/sigma_f)**2)**2)
