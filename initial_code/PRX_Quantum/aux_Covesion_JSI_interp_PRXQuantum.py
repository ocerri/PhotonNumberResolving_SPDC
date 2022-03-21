import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import leastsq, curve_fit
from scipy.interpolate import griddata
import matplotlib.tri as tri
import csv
#from JSI_fit_funcs import *
np.set_printoptions(suppress=True)

"""
Working in units of nm for wavelength and Hz for frequency.
"""
c=3e17 #nm/s
################################
####### READ DATA #######
################################
idler_wavelengths = []
signal_wavelengths = []
coincidences = []


id_peaks=np.array([1605.064,1595.632,1580.656,1565.824,1552,1547.6224,1543.36,1541.1712,1541.0272,1540.264,1540.192,1539.4,1539.04,1538.7808,1537.6288,1536.2392,1529.32,1519.312,1507.36,1495.84,1485.4])
sig_peaks = np.array([1482.75,1490,1501.6,1514.65,1527.7,1532.05,1536.0665,1537.85,1538.459,1539.271,1539.474,1540.3875,1540.75,1541.475,1542.2,1543.65,1550.9,1562.5,1577,1591.5,1606])
coin_peaks = np.array([55,175,315,500,675,740,750,720,720,700,700,775,700,705,705,695,625,430,190,200,25])



################################
####### INTERPOLATION #######
################################

# grid_x, grid_y = np.mgrid[min(idler_wavelengths):max(idler_wavelengths):1000j, min(signal_wavelengths):max(signal_wavelengths):1000j]
# zi = griddata((idler_wavelengths,signal_wavelengths), coincidences, (grid_x, grid_y),method='nearest')
# print("zi ", np.shape(zi))
#
# size = grid_x.size
# x_1d = grid_x.reshape((1, np.prod(size)))
# y_1d = grid_y.reshape((1, np.prod(size)))
# zi_1d = zi.reshape(np.prod(size))
#
# xdata = np.vstack((x_1d, y_1d))
# ydata = zi_1d
#
# data3d = np.array(list(zip(idler_wavelengths,signal_wavelengths,coincidences)))
# data_id = []
# data_sig = []
# data_coin= []
# for p in data3d[:,:3]:
#     data_id.append(p[0])
#     data_sig.append(p[1])
#     data_coin.append(p[2])
# data_id = np.array(data_id)
# data_sig = np.array(data_sig)
# data_coin = np.array(data_coin)





################################
####### FUNCTIONS ##############
################################

def refractive_index_ppln(wavelength):
    w_in_um = 1e-3*wavelength #convert nm to um
    w2 = np.square(w_in_um)
    out =  1+(2.6734*w2)/(w2 - 0.01764)
    out += (1.2290*w2)/(w2 - 0.05914)
    out += (12.614*w2)/(w2 - 474.6)
    return np.sqrt(out)


def gaussian2D(x, y, x_0, y_0, sigma_x, sigma_y):
    aux =  ((x-x_0)/sigma_x)**2
    aux += ((y-y_0)/sigma_y)**2
    norm = 2 * np.pi * sigma_x * sigma_y
    return np.exp(-aux)/norm


def detector_profile(w_idler, w_signal, sigma_d=53, w_0=1550.):
    return gaussian2D(w_idler, w_signal, w_0, w_0, sigma_d, sigma_d)


def pump_envelope(omega_i, omega_s, omega_0, sigma_p):
    #return gaussian2D(omega_i,omega_s,omega_0, omega_0, sigma_p, sigma_p)
    nu_i = omega_i - omega_0
    nu_s = omega_s - omega_0
    # return np.exp(-(nu_i+nu_s)**2/(sigma_p**2)) #Original line

    # No normalization in this gaussian? Yes we could
    aux = (nu_i+nu_s)/sigma_p
    return np.exp(-0.5*np.square(aux))

def phase_mismatch(w_idler, w_signal, w_0): #nanometers
    w_pump = w_0/2
    n_i = refractive_index_ppln(w_idler)
    n_s = refractive_index_ppln(w_signal)
    n_p = refractive_index_ppln(w_pump) #should be index of refraction @ 770 nm right?
    k_i, k_s, k_p = 2*np.pi/w_idler, 2*np.pi/w_signal, 2*np.pi/w_pump
    return n_p*k_p - n_s*k_s - n_i*k_i #1/nm

def sinc2(omega_i, omega_s, omega_0, gamma, L):
    # No 2pi*c to go from omega to wavelength?
    c = 3*10**8 * 10**9 #nm/s
    delta_k = phase_mismatch(2*np.pi*c/omega_i,2*np.pi*c/omega_s,2*np.pi*c/omega_0)-gamma
#     print(n_i, n_s, n_0)
#     print(2*np.pi*c/omega_i, 2*np.pi*c/omega_s,2*np.pi*c/omega_0,)
#     print(phase_match)
#     L= 10**(-4)*10**9
    aux = L*delta_k/2
    return (np.sin(aux)/aux)**2#np.square(np.sinc(aux))


def cwdm_profile(x, y, sigma_x=13., sigma_y=13.):
    # There is no factor 2 to normalize this one? We could
    return gaussian2D(x, y, 1530, 1550, sigma_x, sigma_y) + gaussian2D(x, y, 1550, 1530, sigma_x, sigma_y)


def spdc_profile(w_idler, w_signal, w_central, L, sigma_p=2*np.pi/(40e-12), gamma= 3.9e-4):
    # sigma_p = 2*np.pi/100*10**(-12)
    c = 3*10**8 * 10**9 #nm/s
    omega_i, omega_s, omega_0 = 2*np.pi*c/w_idler, 2*np.pi*c/w_signal, 2*np.pi*c/w_central
    return sinc2(omega_i, omega_s, omega_0, gamma, L)*pump_envelope(omega_i, omega_s, omega_0,sigma_p)


def joint_spectrum(w_idler, w_signal, gamma, A, sigma_p=2*np.pi/(100*10**(-12)), sigma_d=53., w_central=1540., L= 1e7):
    out =  A*detector_profile(w_idler, w_signal, sigma_d)
    out *= spdc_profile(w_idler, w_signal, w_central, L, sigma_p, gamma)
    return out



def _joint_spectrum(M, *args):
    x, y = M[:,0], M[:,1]
    return joint_spectrum(x,y,*args)
