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
    nu_i = omega_i - omega_0
    nu_s = omega_s - omega_0
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
    c = 3.*10**8 * 10**9 #nm/s
    twoPiC = 2.*np.pi*c
    delta_k = phase_mismatch(twoPiC/omega_i, twoPiC/omega_s, twoPiC/omega_0) - gamma
    return np.square(np.sinc(0.5*L*delta_k))


def cwdm_profile(x, y, sigma_x=13., sigma_y=13.):
    # There is no factor 2 to normalize this one? We could
    return gaussian2D(x, y, 1530, 1550, sigma_x, sigma_y) + gaussian2D(x, y, 1550, 1530, sigma_x, sigma_y)


def spdc_profile(w_idler, w_signal, w_central, L, sigma_p=2*np.pi/40e-12, gamma=3.9e-4):
    c = 3.*10**8 * 10**9 #nm/s
    twoPiC = 2.*np.pi*c
    omega_i, omega_s, omega_0 = twoPiC/w_idler, twoPiC/w_signal, twoPiC/w_central
    return sinc2(omega_i, omega_s, omega_0, gamma, L)*pump_envelope(omega_i, omega_s, omega_0, sigma_p)


def joint_spectrum(w_idler, w_signal, gamma, A, sigma_d=53., w_central=1540., sigma_p=2e10*np.pi, L=1e7):
    out =  A*detector_profile(w_idler, w_signal, sigma_d)
    out *= spdc_profile(w_idler, w_signal, w_central, L=L, gamma=gamma, sigma_p=sigma_p)
    return out

def joint_spectrum_noisy(w_idler, w_signal, gamma, sigma_noise, A, A_noise, sigma_d=53., w_central=1540.):
    out = joint_spectrum(w_idler, w_signal, gamma, A, sigma_d, w_central)
    out += gaussian2D(w_idler, w_signal, w_central, w_central, sigma_noise, sigma_noise)
    return out
