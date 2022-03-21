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

if __name__ == '__main__':
    with open('2D_Covesion_JSI_20211030.csv') as csvfile:
        filereader = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in filereader:
            if i==0:
                signal_wavelengths_row=np.array(row[1:-1]).astype(float)
            else:
                for j in range(len(row)-2):
                    if len(row[j+1])>0:
                        signal_wavelengths.append(signal_wavelengths_row[j])
                        idler_wavelengths.append(row[0])
                        coincidences.append(row[j+1])
            i +=1

signal_wavelengths = np.array(signal_wavelengths, dtype = np.float)
idler_wavelengths = np.array(idler_wavelengths, dtype = np.float)
coincidences = np.array(coincidences, dtype = np.float)


id_peaks=np.array([1605.064,1595.632,1580.656,1565.824,1552,1547.6224,1543.36,1541.1712,1541.0272,1540.264,1540.192,1539.4,1539.04,1538.7808,1537.6288,1536.2392,1529.32,1519.312,1507.36,1495.84,1485.4])
sig_peaks = np.array([1482.75,1490,1501.6,1514.65,1527.7,1532.05,1536.0665,1537.85,1538.459,1539.271,1539.474,1540.3875,1540.75,1541.475,1542.2,1543.65,1550.9,1562.5,1577,1591.5,1606])
coin_peaks = np.array([55,175,315,500,675,740,750,720,720,700,700,775,700,705,705,695,625,430,190,200,25])



################################
####### INTERPOLATION #######
################################

grid_x, grid_y = np.mgrid[min(idler_wavelengths):max(idler_wavelengths):1000j, min(signal_wavelengths):max(signal_wavelengths):1000j]
zi = griddata((idler_wavelengths,signal_wavelengths), coincidences, (grid_x, grid_y),method='nearest')
print("zi ", np.shape(zi))

size = grid_x.size
x_1d = grid_x.reshape((1, np.prod(size)))
y_1d = grid_y.reshape((1, np.prod(size)))
zi_1d = zi.reshape(np.prod(size))

xdata = np.vstack((x_1d, y_1d))
ydata = zi_1d

data3d = np.array(list(zip(idler_wavelengths,signal_wavelengths,coincidences)))
data_id = []
data_sig = []
data_coin= []
for p in data3d[:,:3]:
    data_id.append(p[0])
    data_sig.append(p[1])
    data_coin.append(p[2])
data_id = np.array(data_id)
data_sig = np.array(data_sig)
data_coin = np.array(data_coin)





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


print("")
print("refractive index @ 1540 nm: ",refractive_index_ppln(1540))
print("refractive index @ 770 nm: ",refractive_index_ppln(770))
print("")




#################
## FIT ###
#################
num_points=1200

id_wavelengths = np.linspace(1480, 1610, num_points)
sig_wavelengths = np.linspace(1480, 1610, num_points)


gamma_guess = 3.9e-4
A_guess = 7481585.54714451
p_guess=[gamma_guess, A_guess]#,sigma_p_guess]
#if we want to float L, should fix sigma_p, because both have effects on width (can't optimize)
#vice versa, if we want to float sigma_p, should fix L
params, pcov = scipy.optimize.curve_fit(_joint_spectrum,data3d[:,:2], data3d[:,2], p_guess)

params = np.array(params)
params = np.abs(params)


##################################
## PARAMETERS FOR PLOTS #####
##################################

gamma, A = params
L = 1e7 #1 cm in nm
sigma_p = 2*np.pi/(100*10**(-12))


print("")
print("PARAMETERS")
print("Gamma = n/Lambda: ", gamma)
print("A: ", A)
print("sigma_p: ", sigma_p)
print("L (m): ", L*10**(-9))
print("")
print("")


#################
## MAKE MATRICES ###
#################
# gamma = 5*10**6
# A = 103192.62471654158
X,Y = np.meshgrid(id_wavelengths,sig_wavelengths)
detector_mat = detector_profile(X, Y)
pump_mat = pump_envelope(2*np.pi*c/X,2*np.pi*c/ Y,2*np.pi*c/1540, sigma_p)
sinc2_mat = sinc2(2*np.pi*c/X,2*np.pi*c/ Y,2*np.pi*c/1540, gamma, L)
JSI_mat = joint_spectrum(X, Y, *params)
cwdm_mat = cwdm_profile(X, Y)

JSI_mat_cwdm=JSI_mat*cwdm_mat
JSI_mat_cwdm=JSI_mat_cwdm*np.max(JSI_mat)/np.max(JSI_mat_cwdm)


#################
## SVD ###
#################


num_points_svd= 5000
id_wavelengths_svd = np.linspace(1510, 1570, num_points_svd)
sig_wavelengths_svd = np.linspace(1510, 1570, num_points_svd)
X_svd,Y_svd = np.meshgrid(id_wavelengths_svd,sig_wavelengths_svd)
JSI_mat_svd = joint_spectrum(X_svd,Y_svd , *params)
cwdm_mat_svd = cwdm_profile(X_svd,Y_svd )


JSI_mat_cwdm_svd=JSI_mat_svd*cwdm_mat_svd
JSI_mat_cwdm_svd=JSI_mat_cwdm_svd*np.max(JSI_mat_svd)/np.max(JSI_mat_cwdm_svd)


u_cwdm, s_cwdm, vh_cwdm = np.linalg.svd(JSI_mat_cwdm_svd)#/np.sum(signal1_idler_twofold))
svd_cwdm = s_cwdm/np.sum(s_cwdm)
print(svd_cwdm)


u, s, vh = np.linalg.svd(JSI_mat)#/np.sum(signal1_idler_twofold))
svd= s/np.sum(s)



## SAVE SVD ###
np.savetxt("JSI_paper_prx_quantum.txt",svd_cwdm)




# sum = 0
# for l in svd_cwdm:
#     sum+=l**2
sum = np.sum(svd_cwdm**2)
K_cwdm= 1/sum
print("svd cwdm")
print(svd_cwdm)
print("K cwdm= ", K_cwdm)
print()


#num_lambdas = 400
lambda_index = np.matrix.flatten(np.argwhere(svd_cwdm>1e-4))
svd_cwdm = svd_cwdm[lambda_index]#[:len(lambda_index)]
x_tick_labels = [r'$\lambda_{{{:d}}}$'.format(i+1) for i in range(len(svd_cwdm))]
#print(x_tick_labels)




#################
## PLOTTING ###
#################


### PLOT PHASE MISMATCH ###
fig, ax = plt.subplots(1,1)
cp=ax.contourf(X,Y,phase_mismatch(2*np.pi*c/X,2*np.pi*c/Y, 2*np.pi*c/1540 ), levels = 50)
fig.colorbar(cp)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Wavelength (nm)")
ax.set_title("Phase mismatch")

# From this I find what the m/Gamma should be for 1540 nm (=3.9e-4)

#### PLOT JSIs###

fontsize = 10
fontsize_x = 9
fontsize_y = 9
labelsize = 8
#Plotting modes

levels = np.arange(0,1.005,0.005)
ticks = np.arange(0,1.25,0.25)
fig, ax = plt.subplots(1,1)
cp=ax.contourf(id_wavelengths,sig_wavelengths, JSI_mat/np.max(JSI_mat) ,80,levels=levels)
sc = ax.scatter(id_peaks,sig_peaks, facecolors="None",c=coin_peaks/np.max(coin_peaks),cmap = 'viridis', s = 30, alpha = 1)
ax.set_title("Measured JSI",fontsize=fontsize)
ax.set_xlabel("Wavelength (nm)",fontsize=fontsize_x)
ax.set_ylabel("Wavelength (nm)",fontsize=fontsize_y)
cbar = fig.colorbar(cp,ax = ax)#, levels = levels)
cbar.set_ticks(ticks)
#plt.clim(0,1)



cbar_arr=[]
fig, axs = plt.subplots(3,2, figsize=(9,6))
ax_text_loc = (-0.01,1.11)
ax = axs[0][0]
cp = ax.contourf(id_wavelengths,sig_wavelengths, sinc2_mat/np.max(sinc2_mat),80,levels=levels)
ax.set_xlabel("Wavelength (nm)",fontsize=fontsize_x)
ax.set_ylabel("Wavelength (nm)",fontsize=fontsize_y)
ax.set_title("Phase matching amplitude", fontsize=fontsize)
cbar = fig.colorbar(cp,ax = ax)
cbar.ax.tick_params(labelsize=labelsize)
cbar.set_ticks(ticks)
ax.annotate('a)',
            xy=ax_text_loc, xycoords='axes fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=fontsize)
ax = axs[0][1]
cp = ax.contourf(id_wavelengths,sig_wavelengths, pump_mat/np.max(pump_mat),80,levels=levels)
ax.set_xlabel("Wavelength (nm)",fontsize=fontsize_x)
ax.set_ylabel("Wavelength (nm)",fontsize=fontsize_y)
ax.set_title("Pump spectral amplitude", fontsize=fontsize)
cbar = fig.colorbar(cp,ax = ax)
cbar.ax.tick_params(labelsize=labelsize)
cbar.set_ticks(ticks)
ax.annotate('b)',
            xy=ax_text_loc, xycoords='axes fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=fontsize)
ax = axs[1][0]
cp = ax.contourf(id_wavelengths,sig_wavelengths, detector_mat/np.max(detector_mat),80,levels=levels)
ax.set_xlabel("Wavelength (nm)",fontsize=fontsize_x)
ax.set_ylabel("Wavelength (nm)",fontsize=fontsize_y)
ax.set_title("Detector response",fontsize=fontsize)
cbar = fig.colorbar(cp,ax = ax)
cbar.ax.tick_params(labelsize=labelsize)
cbar.set_ticks(ticks)
ax.annotate('c)',
            xy=ax_text_loc, xycoords='axes fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=fontsize)
ax = axs[1][1]
cp=ax.contourf(id_wavelengths,sig_wavelengths, JSI_mat/np.max(JSI_mat) ,80,levels=levels)
sc = ax.scatter(id_peaks,sig_peaks, facecolors=None,c=coin_peaks/np.max(coin_peaks),cmap = 'viridis', s = 10, alpha = 1)
ax.set_title("Measured JSI",fontsize=fontsize)
ax.set_xlabel("Wavelength (nm)",fontsize=fontsize_x)
ax.set_ylabel("Wavelength (nm)",fontsize=fontsize_y)
cbar = fig.colorbar(cp,ax = ax)
cbar.ax.tick_params(labelsize=labelsize)
cbar.set_ticks(ticks)
ax.annotate('d)',
            xy=ax_text_loc, xycoords='axes fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=fontsize)
ax = axs[2][0]
cp=ax.contourf(id_wavelengths,sig_wavelengths, cwdm_mat/np.max(cwdm_mat) ,80,levels=levels)
ax.set_title("Filter response",fontsize=fontsize)
ax.set_xlabel("Wavelength (nm)",fontsize=fontsize_x)
ax.set_ylabel("Wavelength (nm)",fontsize=fontsize_y)
cbar = fig.colorbar(cp,ax = ax)
cbar.ax.tick_params(labelsize=labelsize)
cbar.set_ticks(ticks)
ax.annotate('e)',
            xy=ax_text_loc, xycoords='axes fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=fontsize)
ax = axs[2][1]
cp = ax.contourf(id_wavelengths,sig_wavelengths, JSI_mat_cwdm/np.max(JSI_mat_cwdm) ,80,levels=levels)
ax.set_title("JSI of experiment",fontsize=fontsize)
ax.set_xlabel("Wavelength (nm)", fontsize=fontsize_x)
ax.set_ylabel("Wavelength (nm)",fontsize=fontsize_y)
cbar = fig.colorbar(cp,ax = ax)
cbar.ax.tick_params(labelsize=labelsize)
cbar.set_ticks(ticks)
ax.annotate('f)',
            xy=ax_text_loc, xycoords='axes fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=fontsize)
for i in range(3):
    for j in range(2):
        axs[i][j].tick_params(axis='both', which='major', labelsize=labelsize)
        axs[i][j].tick_params(axis='both', which='minor', labelsize=labelsize)
plt.subplots_adjust(left = 0.077, bottom = 0.055, right = 0.765, top = 0.944, wspace = 0.34, hspace =0.374)




###JSI for experiment###

fig, axs = plt.subplots(1,1)
cp = axs.contourf(id_wavelengths_svd,sig_wavelengths_svd, JSI_mat_cwdm_svd/np.max(JSI_mat_cwdm_svd) ,50)
axs.set_title("JSI for Experiment",fontsize=fontsize)
axs.set_xlabel("Wavelength (nm)", fontsize=fontsize_x)
axs.set_ylabel("Wavelength (nm)",fontsize=fontsize_y)
axs.set_xticks([1520,1540,1560])
axs.set_yticks([1520,1540,1560])
z = JSI_mat_cwdm/np.sum(JSI_mat_cwdm)
# ticks = np.linspace(z.min(), z.max(), 3, endpoint=True)
# cbar = fig.colorbar(cp,ax = axs, format='%.1f', ticks = ticks)
# cbar.ax.tick_params(labelsize=8)
fig.colorbar(cp)
axs.tick_params(axis='both', which='major', labelsize=8)
axs.tick_params(axis='both', which='minor', labelsize=8)


# with plt.rc_context({'image.composite_image': False}):
#     fig.savefig('jsi_model_vertical.pdf',dpi=1000)






## PLOT SVD ###
fig, axs = plt.subplots(1,1)
width = 0.9
axs.plot(lambda_index-0*np.ones(len(lambda_index))/2, svd_cwdm,".k", label = "K = {:.1f}".format(K_cwdm))
axs.bar(lambda_index-0*np.ones(len(lambda_index))/2, svd_cwdm,width,color = "g", alpha = 0.5)#, label = "No Filter (Fit), K = {:2f}".format(K_cwdm))
axs.set_xlabel("Index (s)")
#ax.set_xticks(lambda_index)
#ax.set_xticklabels(x_tick_labels, fontsize = 9)
axs.set_ylabel(r"Eigenvalues $\lambda_s$")
axs.set_title("Schmidt Decomposition")
axs.legend()




plt.show()
