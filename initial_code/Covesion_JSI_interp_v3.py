import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import leastsq, curve_fit
from scipy.interpolate import griddata
import matplotlib.tri as tri
import csv
from JSI_fit_funcs import *
np.set_printoptions(suppress=True)

idler_wavelengths = []
signal_wavelengths = []
coincidences = []

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

fig, ax = plt.subplots(1,1)
sc = ax.scatter(idler_wavelengths,signal_wavelengths, facecolors=None,c=coincidences,cmap = 'viridis', s = 1, alpha = 1)
cbar = fig.colorbar(sc,ax = ax)

id_peaks=[1605.064,1595.632,1580.656,1565.824,1552,1547.6224,1543.36,1541.1712,1541.0272,1540.264,1540.192,1539.4,1539.04,1538.7808,1537.6288,1536.2392,1529.32,1519.312,1507.36,1495.84,1485.4]
sig_peaks = [1482.75,1490,1501.6,1514.65,1527.7,1532.05,1536.0665,1537.85,1538.459,1539.271,1539.474,1540.3875,1540.75,1541.475,1542.2,1543.65,1550.9,1562.5,1577,1591.5,1606]
coin_peaks = [55,175,315,500,675,740,750,720,720,700,700,775,700,705,705,695,625,430,190,200,25]

fig, ax = plt.subplots(1,1)
sc = ax.scatter(id_peaks,sig_peaks, facecolors=None,c=coin_peaks,cmap = 'viridis', s = 5, alpha = 1)
cbar = fig.colorbar(sc,ax = ax)


#Interpolation
grid_x, grid_y = np.mgrid[min(idler_wavelengths):max(idler_wavelengths):1000j, min(signal_wavelengths):max(signal_wavelengths):1000j]
zi = griddata((idler_wavelengths,signal_wavelengths), coincidences, (grid_x, grid_y),method='nearest')
print("zi ", np.shape(zi))

size = grid_x.size
x_1d = grid_x.reshape((1, np.prod(size)))
y_1d = grid_y.reshape((1, np.prod(size)))
zi_1d = zi.reshape(np.prod(size))

xdata = np.vstack((x_1d, y_1d))
ydata = zi_1d

fig, ax = plt.subplots(1,1)
ax.contourf(grid_x,grid_y, zi, levels = 100)
ax.set_xlabel("Idler Wavelengths")
ax.set_ylabel("Signal Wavelengths")



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


num_points=1200


lambda_0 = 1540
omega_0 = 1/lambda_0
id_wavelengths = np.linspace(1480, 1610, num_points)
sig_wavelengths = np.linspace(1480, 1610, num_points)
id_ks,sig_ks = 1/id_wavelengths,1/sig_wavelengths
id_omegas,sig_omegas = id_ks,sig_ks


# #with CWDM
# sig_center_cwdm, id_center_cwdm = 1530, 1550
# cwdm_bandwidth = 13
# sig_min_cwdm, sig_max_cwdm = sig_center_cwdm-cwdm_bandwidth/2, sig_center_cwdm+cwdm_bandwidth/2
# id_min_cwdm, id_max_cwdm = id_center_cwdm-cwdm_bandwidth/2, id_center_cwdm+cwdm_bandwidth/2
# id_wavelengths_cwdm = np.linspace(id_min_cwdm,id_max_cwdm, num_points)
# sig_wavelengths_cwdm = np.linspace(sig_min_cwdm,sig_max_cwdm, num_points)
# id_ks_cwdm,sig_ks_cwdm = 1/id_wavelengths_cwdm,1/sig_wavelengths_cwdm
# id_omegas_cwdm,sig_omegas_cwdm = id_ks_cwdm,sig_ks_cwdm


#CURVE FITTING
def refractive_index_ppln(wavelength):
    return np.sqrt(1+(2.6734*wavelength**2)/(wavelength**2-0.01764)+(1.2290*wavelength**2)/(wavelength**2 - 0.05914)+(12.614*wavelength**2)/(wavelength**2-474.6))

def gaussian2D(x,y,x_0, y_0, sigma_x, sigma_y):
    A = 1/(np.pi*2 * sigma_x * sigma_y)
    return A*np.exp(-((((x-x_0)/(sigma_x))**2 + ((y-y_0)/(sigma_y))**2 )))

def detector_profile(idler_wavelengths,signal_wavelengths, sigma_d):
    return gaussian2D(idler_wavelengths,signal_wavelengths,1550, 1550, sigma_d, sigma_d)

def pump_envelope(omega_i, omega_s, omega_0,sigma_p):
    #return gaussian2D(omega_i,omega_s,omega_0, omega_0, sigma_p, sigma_p)
    nu_i = omega_i- omega_0
    nu_s = omega_s - omega_0
    return np.exp(-(nu_i+nu_s)**2/(sigma_p**2))

def sinc2(omega_i, omega_s, omega_0,gamma):
    n_i = refractive_index_ppln(1/omega_i)
    n_s = refractive_index_ppln(1/omega_s)
    n_0 = refractive_index_ppln(1/omega_0)
    L = 0.01*10**(-2)*10**9
    return np.sinc(L*np.pi*(n_i*omega_i + n_s*omega_s -2*n_0*omega_0 +gamma))**2#np.sinc((nu_i+nu_s)/(2*gamma))**2

def cwdm_profile(x,y):
    sigma_x, sigma_y  = 13,13
    return gaussian2D(x,y,1530, 1550, sigma_x, sigma_y) + gaussian2D(x,y,1550, 1530, sigma_x, sigma_y)

def spdc_profile(idler_wavelengths,signal_wavelengths, central_wavelength, gamma):#, sigma_p):
    sigma_p = 3*10**(8)*100*10**(-12)
    omega_i, omega_s, omega_0 = 1/idler_wavelengths, 1/signal_wavelengths, 1/central_wavelength
    return sinc2(omega_i, omega_s, omega_0,gamma)*pump_envelope(omega_i, omega_s, omega_0,sigma_p)


def joint_spectrum(idler_wavelengths,signal_wavelengths,gamma, A):
    sigma_d = 53
    central_wavelength = 1540
    return A*detector_profile(idler_wavelengths,signal_wavelengths, sigma_d)*spdc_profile(idler_wavelengths,signal_wavelengths, central_wavelength, gamma)

def joint_spectrum_noisy(idler_wavelengths,signal_wavelengths, gamma, sigma_noise, A, A_noise):
    sigma_d = 53
    central_wavelength = 1540
    return joint_spectrum(idler_wavelengths,signal_wavelengths, gamma,A)+A_noise*gaussian2D(idler_wavelengths,signal_wavelengths,central_wavelength,central_wavelength, sigma_noise, sigma_noise)

def _joint_spectrum(M, *args):
    x, y = M[:,0], M[:,1]
    return joint_spectrum(x,y,*args)


def _joint_spectrum_interp(M, *args):
    x, y = M[0], M[1]
    return joint_spectrum(x,y,*args)

def _joint_spectrum_noisy(M, *args):
    x, y = M[:,0], M[:,1]
    return joint_spectrum_noisy(x,y,*args)







gamma = 0#1/(15*10**(-6))*10**(-9)#50*10**(-9)/(2*np.pi)#1/(1540)-1/(1543)
sigma_d = 53
A = 7481585.54714451
sigma_p = 10**(-9)/(3*10**8 * 100*10**(-12))
p_guess=[gamma, A]

params, pcov = scipy.optimize.curve_fit(_joint_spectrum,data3d[:,:2], data3d[:,2], p_guess)

params = np.array(params)
params = np.abs(params)
#params = np.array([5.829787689737916*10**(-7),11678008.661648719])
print(params[0],params[1])
print("Lambda (um): ", (1/params[0])*10**(-9)*10**(6))

detector_mat = np.zeros((len(sig_wavelengths),len(id_wavelengths)))
pump_mat = np.zeros((len(sig_wavelengths),len(id_wavelengths)))
sinc2_mat = np.zeros((len(sig_wavelengths),len(id_wavelengths)))
SPDC_mat = np.zeros((len(sig_wavelengths),len(id_wavelengths)))
JSI_mat = np.zeros((len(sig_wavelengths),len(id_wavelengths)))#joint_spectrum(X_lambda, Y_lambda, *p)
JSI_mat_noisy = np.zeros((len(sig_wavelengths),len(id_wavelengths)))
cwdm_mat =np.zeros((len(sig_wavelengths),len(id_wavelengths)))
for i in range(len(id_wavelengths)):
    for j in range(len(sig_wavelengths)):
        if i%100==0 and j%100==0:
            print(i,j)
        detector_mat[j][i] = detector_profile(id_wavelengths[i], sig_wavelengths[j], sigma_d)
        pump_mat[j][i] = pump_envelope(1/id_wavelengths[i], 1/sig_wavelengths[j],1/lambda_0, sigma_p)
        sinc2_mat[j][i] = sinc2(1/id_wavelengths[i], 1/sig_wavelengths[j], 1/lambda_0,params[0])
        #SPDC_mat[j][i]=spdc_profile(id_wavelengths[i], sig_wavelengths[j], lambda_0, params[0])#, params[3])
        JSI_mat[j][i]=joint_spectrum(id_wavelengths[i], sig_wavelengths[j], *params)
        #JSI_mat_noisy[j][i]=joint_spectrum_noisy(id_wavelengths[i], sig_wavelengths[j], *params_noisy)
        cwdm_mat[j][i] = cwdm_profile(id_wavelengths[i], sig_wavelengths[j])

JSI_mat_cwdm=JSI_mat*cwdm_mat#/np.sum(cwdm_mat)
JSI_mat_cwdm=JSI_mat_cwdm*np.max(JSI_mat)/np.max(JSI_mat_cwdm)


u_cwdm, s_cwdm, vh_cwdm = np.linalg.svd(JSI_mat_cwdm)#/np.sum(signal1_idler_twofold))
svd_cwdm = s_cwdm/np.sum(s_cwdm)
print(svd_cwdm)


u, s, vh = np.linalg.svd(JSI_mat)#/np.sum(signal1_idler_twofold))
svd= s/np.sum(s)





sum = 0
for l in svd_cwdm:
    sum+=l**2
K_cwdm= 1/sum
print("svd cwdm")
print(svd_cwdm)
print("K cwdm= ", K_cwdm)
print()



lambda_index = range(60)
svd_cwdm = svd_cwdm[:len(lambda_index)]
x_tick_labels = [r'$\lambda_{{{:d}}}$'.format(i+1) for i in lambda_index]
#print(x_tick_labels)



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
#sc = ax.scatter(idler_wavelengths,signal_wavelengths, facecolors=None,c=coincidences/np.sum(coincidences),cmap = 'viridis', s = 2, alpha = 1)
#sc = ax.scatter(idler_wavelengths,signal_wavelengths, facecolors="None",edgecolors = "w", s = 1, alpha = 0.25)
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
#ax.ticklabel_format(axis="z", style="sci", scilimits=(0,0))

ax = axs[0][1]
cp = ax.contourf(id_wavelengths,sig_wavelengths, pump_mat/np.max(pump_mat),80,levels=levels)
ax.set_xlabel("Wavelength (nm)",fontsize=fontsize_x)
ax.set_ylabel("Wavelength (nm)",fontsize=fontsize_y)
ax.set_title("Pump spectral amplitude", fontsize=fontsize)
cbar = fig.colorbar(cp,ax = ax)
cbar.ax.tick_params(labelsize=labelsize)
cbar.set_ticks(ticks)
#ax.ticklabel_format(axis="z", style="sci", scilimits=(0,0))
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
#ax.ticklabel_format(axis="z", style="sci", scilimits=(0,0))
ax.annotate('c)',
            xy=ax_text_loc, xycoords='axes fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=fontsize)

ax = axs[1][1]
cp=ax.contourf(id_wavelengths,sig_wavelengths, JSI_mat/np.max(JSI_mat) ,80,levels=levels)
sc = ax.scatter(id_peaks,sig_peaks, facecolors=None,c=coin_peaks/np.max(coin_peaks),cmap = 'viridis', s = 10, alpha = 1)

#sc = ax.scatter(idler_wavelengths,signal_wavelengths, facecolors=None,c=coincidences/np.sum(coincidences),cmap = 'viridis', s = 1, alpha = 1)
ax.set_title("Measured JSI",fontsize=fontsize)
ax.set_xlabel("Wavelength (nm)",fontsize=fontsize_x)
ax.set_ylabel("Wavelength (nm)",fontsize=fontsize_y)
cbar = fig.colorbar(cp,ax = ax)
cbar.ax.tick_params(labelsize=labelsize)
cbar.set_ticks(ticks)
#ax.ticklabel_format(axis="z", style="sci", scilimits=(0,0))
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
#ax.ticklabel_format(axis="z", style="sci", scilimits=(0,0))
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
#ax.ticklabel_format(axis="z", style="sci", scilimits=(0,0))
ax.annotate('f)',
            xy=ax_text_loc, xycoords='axes fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=fontsize)


for i in range(3):
    for j in range(2):
        axs[i][j].tick_params(axis='both', which='major', labelsize=labelsize)
        axs[i][j].tick_params(axis='both', which='minor', labelsize=labelsize)
plt.subplots_adjust(left = 0.077, bottom = 0.055, right = 0.765, top = 0.944, wspace = 0.34, hspace =0.374)


# id_wavelengths = np.linspace(1529, 1539, np.shape(JSI_mat_cwdm)[1])
# sig_wavelengths = np.linspace(1529, 1539, np.shape(JSI_mat_cwdm)[0])
fig, axs = plt.subplots(1,1)
cp = axs.contourf(id_wavelengths,sig_wavelengths, JSI_mat_cwdm/np.max(JSI_mat_cwdm) ,50)
#cp = axs[1][1].contourf(grid_x,grid_y,zi)
axs.set_title("JSI for Experiment",fontsize=fontsize)
axs.set_xlabel("Wavelength (nm)", fontsize=fontsize_x)
axs.set_ylabel("Wavelength (nm)",fontsize=fontsize_y)
axs.set_xticks([1520,1540,1560])
axs.set_yticks([1520,1540,1560])
z = JSI_mat_cwdm/np.sum(JSI_mat_cwdm)
ticks = np.linspace(z.min(), z.max(), 3, endpoint=True)
cbar = fig.colorbar(cp,ax = axs, format='%.1f', ticks = ticks)
cbar.ax.tick_params(labelsize=8)
axs.tick_params(axis='both', which='major', labelsize=8)
axs.tick_params(axis='both', which='minor', labelsize=8)


# with plt.rc_context({'image.composite_image': False}):
#     fig.savefig('jsi_model_vertical.pdf',dpi=1000)


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




# file = open("20220110_JSI_Fit_SVD_cwdm_interp_gaussian.txt","w")
# for l in svd_cwdm:
#     file.write(str(l)+"\n")
# file.close()
#





plt.show()
