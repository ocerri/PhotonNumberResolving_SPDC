import numpy as np

#################
### To be checked
#################

def refractive_index_ppln(wavelength):
    w_in_um = 1e-3*wavelength
    w2 = np.square(w_in_um)
    out =  1+(2.6734*w2)/(w2 - 0.01764)
    out += (1.2290*w2)/(w2 - 0.05914)
    out += (12.614*w2)/(w2 - 474.6)
    return np.sqrt(out)


def gaussian2D(x, y, x_0, y_0, sigma_x, sigma_y):
    aux =  np.square((x-x_0)/sigma_x)
    aux += np.square((y-y_0)/sigma_y)
    norm = 2 * np.pi * sigma_x * sigma_y
    return np.exp(-aux)/norm


def detector_profile(w_idler, w_signal, sigma_d, w_0=1550.):
    return gaussian2D(w_idler, w_signal, w_0, w_0, sigma_d, sigma_d)


def pump_envelope(omega_i, omega_s, omega_0, sigma_p):
    #return gaussian2D(omega_i,omega_s,omega_0, omega_0, sigma_p, sigma_p)
    nu_i = omega_i - omega_0
    nu_s = omega_s - omega_0
    # return np.exp(-(nu_i+nu_s)**2/(sigma_p**2)) #Original line

    # No normalization in this gaussian? Yes we could
    aux = (nu_i+nu_s)/sigma_p
    return np.exp(-0.5*np.square(aux))


def sinc2(omega_i, omega_s, omega_0, gamma):
    # No 2pi*c to go from omega to wavelength?
    n_i = refractive_index_ppln(1/omega_i)
    n_s = refractive_index_ppln(1/omega_s)
    n_0 = refractive_index_ppln(1/omega_0)
    # L = 0.01*10**(-2)*10**9
    L = 100000.
    # return np.sinc(L*np.pi*(n_i*omega_i + n_s*omega_s -2*n_0*omega_0 +gamma))**2#np.sinc((nu_i+nu_s)/(2*gamma))**2
    aux = L*np.pi*(n_i*omega_i + n_s*omega_s -2*n_0*omega_0 - gamma)
    # how to normalize that?
    return np.square(np.sinc(aux))


def cwdm_profile(x, y, sigma_x=13., sigma_y=13.):
    # There is no factor 2 to normalize this one? We could
    return gaussian2D(x, y, 1530, 1550, sigma_x, sigma_y) + gaussian2D(x, y, 1550, 1530, sigma_x, sigma_y)


def spdc_profile(w_idler, w_signal, w_central, gamma, sigma_p=0.03):
    # sigma_p = 3*10**(8)*100*10**(-12)
    omega_i, omega_s, omega_0 = 1./w_idler, 1./w_signal, 1./w_central
    return sinc2(omega_i, omega_s, omega_0, gamma)*pump_envelope(omega_i, omega_s, omega_0,sigma_p)


def joint_spectrum(w_idler, w_signal, gamma, A, sigma_d=53., w_central=1540., sigma_p=0.03):
    out =  A*detector_profile(w_idler, w_signal, sigma_d)
    out *= spdc_profile(w_idler, w_signal, w_central, gamma, sigma_p)
    return out

def _joint_spectrum(M, *args):
    x, y = M[:,0], M[:,1]
    return joint_spectrum(x,y,*args)


def joint_spectrum_noisy(w_idler, w_signal, gamma, sigma_noise, A, A_noise, sigma_d=53., w_central=1540.):
    out = joint_spectrum(w_idler, w_signal, gamma, A, sigma_d, w_central)
    out += gaussian2D(w_idler, w_signal, w_central, w_central, sigma_noise, sigma_noise)
    return out

def _joint_spectrum_noisy(M, *args):
    x, y = M[:,0], M[:,1]
    return joint_spectrum_noisy(x,y,*args)
