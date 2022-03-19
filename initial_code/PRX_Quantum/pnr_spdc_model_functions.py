import math
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin
from scipy.io import loadmat
import numpy as np
from scipy.optimize import curve_fit, root, fsolve
from scipy.interpolate import interp1d
from scipy.stats import poisson


###################
## GET COEFFICIENTS
###################
def nCk(n,k):
    f = math.factorial
    return f(n) / (f(k) * f(n-k))

def pnk(n,k):
    return nCk(n,k)/(2**n)

def P_eta_N_k_n(eta,N,k,n):
    sum = 0
    for j in range(k+1):
        sum+=(-1)**j * nCk(k,j)*((1-eta)+(k-j)*eta/N)**n
    return sum*nCk(N,k)

def muCoeff(mu,n):
    return mu**n/(1+mu)**(n+1)

def binaryEffCoeff(eta,n):
    return 1-(1-eta)**n
def PnrEffCoeff(eta,n):
    return eta*(1-eta)**(n-1)


def stirling(n,k):
    n1=n
    k1=k
    if n<=0:
        return 1

    elif k<=0:
        return 0

    elif (n==0 and k==0):
        return -1

    elif n!=0 and n==k:
        return 1

    elif n<k:
        return 0

    else:
        temp1=stirling(n1-1,k1)
        temp1=k1*temp1
        return (k1*(stirling(n1-1,k1)))+stirling(n1-1,k1-1)

###################
## MODEL FUNCTIONS
###################

#Pdark_i = probability of getting at least 1 dark count on idler detector
#Pdark_s1 = probability of getting at least 1 dark count on signal 1 detector
#Pdark_s2 = probability of getting at least 1 dark count on signal 2 detector

#P(nd=1) = P(nd>=1)/(1+P(nd>=1))



def Ri_binary_analytic(mu, eta_i, lambdas=[1]):
    term = 1
    for l in lambdas:
        m = l*mu
        term = term*(1/(1+eta_i*m))
    total = 1-term
    # if len(total)>1:
    #     if total.any() < 0:
    #         print("negative Ri_binary_analytic: ",total)
    #         return 0
    # elif total < 0:
    #     print("negative Ri_binary_analytic: ",total)
    #     return 0
    return total

def Ri_pnr_analytic(mu, eta_i, k=10, lambdas = [1]):
    term1 = 1
    term2 = 1
    for l in lambdas:
        m = l*mu
        term1 *= 2**k/(2**k + (2**k-1)*m*eta_i)
        term2 *= 1/(1+m*eta_i)
    total = 2**k*(term1- term2)
    # if len(total)>1:
    #     if total.any() < 0:
    #         print("negative Ri_pnr_analytic: ",total)
    #         return 0
    # elif total < 0:
    #     print("negative Ri_pnr_analytic: ",total)
    #     return 0
    return total



#eta*mu/(1+mu*eta)^2 is the limit k--> infty

def Rsj_analytic(mu, eta_sj, lambdas=[1]):
    term = 1
    for l in lambdas:
        m = l*mu
        term = term*(2/(2+eta_sj*m))
    total =  1-term
    # if len(total)>1:
    #     if total.any() < 0:
    #         print("negative Rsj_analytic: ",total)
    #         return 0
    # elif total < 0:
    #     print("negative Rsj_analytic: ",total)
    #     return 0
    return total


def Rs1s2_analytic(mu,eta_s1,eta_s2, lambdas=[1]):
    term_s1 = 1
    term_s2 = 1
    term_s1s2 = 1
    for l in lambdas:
        m = l*mu
        term_s1 = term_s1*(2/(2+eta_s1*m))
        term_s2 = term_s2*(2/(2+eta_s2*m))
        term_s1s2 = term_s1s2*(2/(2+m*(eta_s1+eta_s2)))
    total =  1-term_s1 -term_s2 + term_s1s2
    # if len(total)>1:
    #     if total.any() < 0:
    #         print("negative Rs1s2_analytic: ",total)
    #         return 0
    # elif total < 0:
    #     print("negative Rs1s2_analytic: ",total)
    #     return 0
    return total


def Risj_binary_analytic(mu,eta_sj,eta_i, lambdas=[1]):
    term_s = 1
    term_i = 1
    term_si = 1
    for l in lambdas:
        m = l*mu
        term_s = term_s*(2/(2+eta_sj*m))
        term_i = term_i*(1/(1+eta_i*m))
        term_si = term_si*(2/(2+m*(eta_sj+2*eta_i-eta_sj*eta_i)))
    total = 1-term_s -term_i + term_si
    # if len(total)>1:
    #     if total.any() < 0:
    #         print("negative Risj_binary_analytic: ",total)
    #         return 0
    # elif total < 0:
    #     print("negative Risj_binary_analytic: ",total)
    #     return 0
    return total

def Risj_pnr_analytic(mu, eta_sj, eta_i, k=10, lambdas = [1]):
    term1 = 1
    term2 = 1
    term3 = 1
    term4 = 1
    for l in lambdas:
        m = l*mu
        term1*= 2**k/(2**k + (2**k -1)*m*eta_i)
        term2*= 2**(k+1)/(m*eta_sj*(2**k - (2**k-1)*eta_i)+2*(2**k + (2**k - 1)*m*eta_i))
        term3*=1/(1+m*eta_i)
        term4*= 2/(2+2*m*eta_i + eta_sj * m*(1-eta_i))
    total = 2**k*(term1 -term2 -term3 + term4)
    # if len(total)>1:
    #     if total.any() < 0:
    #         print("negative Risj_pnr_analytic: ",total)
    #         return 0
    # elif total < 0:
    #     print("negative Risj_pnr_analytic: ",total)
    #     return 0
    return total

def Ris1s2_binary_analytic(mu, eta_s1,eta_s2, eta_i, lambdas=[1]):
    term_s1 = 1
    term_s2 = 1
    term_i = 1
    term_s1i = 1
    term_s2i = 1
    term_s1s2 = 1
    term_s1s2i=1
    for l in lambdas:
        m = l*mu
        term_s1 = term_s1*(2/(2+eta_s1*m))
        term_s2 = term_s2*(2/(2+eta_s2*m))
        term_i = term_i*(1/(1+eta_i*m))
        term_s1s2 = term_s1s2*(2/(2+m*(eta_s1+eta_s2)))
        term_s1i = term_s1i*(2/(2+m*(eta_s1+2*eta_i-eta_s1*eta_i)))
        term_s2i = term_s2i*(2/(2+m*(eta_s2+2*eta_i-eta_s2*eta_i)))
        term_s1s2i=term_s1s2i*(2/(2+m*(eta_s1+eta_s2 + 2*eta_i - eta_i*eta_s1 - eta_i*eta_s2)))
    total =  1 - term_s1 - term_s2 - term_i + term_s1i + term_s2i + term_s1s2 - term_s1s2i
    # if len(total)>1:
    #     if total.any() < 0:
    #         print("negative Ris1s2_binary_analytic: ",total)
    #         return 0
    # elif total < 0:
    #     print("negative Ris1s2_binary_analytic: ",total)
    #     return 0
    return total


def Ris1s2_pnr_analytic(mu, eta_s1,eta_s2, eta_i, k=10,lambdas=[1]):
    term1 = 1
    term2 = 1
    term3 = 1
    term4 = 1
    term5 = 1
    term6 = 1
    term7=1
    term8=1
    for l in lambdas:
        m = l*mu
        term1*=2**k/(2**k + (2**k -1)*m*eta_i)
        term2*=2**(k+1)/(m*eta_s1*(2**k - (2**k - 1)*eta_i)+2*(2**k + (2**k - 1)*m*eta_i))
        term3*=2**(k+1)/(m*eta_s2*(2**k - (2**k - 1)*eta_i)+2*(2**k + (2**k - 1)*m*eta_i))
        term4*=2**(k+1)/(m*(eta_s1+eta_s2)*(2**k - (2**k - 1)*eta_i)+2*(2**k + (2**k - 1)*m*eta_i))
        term5*=1/(1+m*eta_i)
        term6*=2/(2+2*m*eta_i + eta_s1*m*(1-eta_i))
        term7*=2/(2+2*m*eta_i + eta_s2*m*(1-eta_i))
        term8*=2/(2+2*m*eta_i + (eta_s1+eta_s2)*m*(1-eta_i))
    total = 2**k*(term1-term2-term3+term4-term5+term6+term7-term8)
    # if len(total)>1:
    #     if total.any() < 0:
    #         print("negative Ris1s2_pnr_analytic: ",total)
    #         return 0
    # elif total < 0:
    #     print("negative Ris1s2_pnr_analytic: ",total)
    #     return 0
    return total


def g2_thermal(mu, eta_s1,eta_s2, lambdas=[1]):
    Rs1 = Rsj_analytic(mu,eta_s1,lambdas=lambdas)
    Rs2 = Rsj_analytic(mu,eta_s2,lambdas=lambdas)
    Rs1s2 = Rs1s2_analytic(mu,eta_s1,eta_s2, lambdas = lambdas)
    return  Rs1s2/(Rs1*Rs2)

def g2_0_binary(mu, eta_s1,eta_s2, eta_i, lambdas=[1]):
    Ris1 = Risj_binary_analytic(mu,eta_s1,eta_i,lambdas=lambdas)
    Ris2 = Risj_binary_analytic(mu,eta_s2,eta_i,lambdas=lambdas)
    Ris1s2=Ris1s2_binary_analytic(mu, eta_s1,eta_s2, eta_i,lambdas=lambdas)
    Ri = Ri_binary_analytic(mu,eta_i,lambdas=lambdas)
    return Ris1s2*Ri/(Ris1*Ris2)

def g2_0_pnr(mu, eta_i,eta_s1,eta_s2,  k=10, lambdas=[1]):
    Ris1 = Risj_pnr_analytic(mu,eta_s1,eta_i,k, lambdas=lambdas)
    Ris2 = Risj_pnr_analytic(mu,eta_s2,eta_i,k,lambdas=lambdas)
    Ris1s2=Ris1s2_pnr_analytic(mu, eta_s1,eta_s2, eta_i,k,lambdas=lambdas)
    Ri = Ri_pnr_analytic(mu,eta_i,k,lambdas=lambdas)
    return Ris1s2*Ri/(Ris1*Ris2)


def g_2_binary_to_pnr_ratio(mu,eta_s1,eta_s2,eta_i, k=10, lambdas = [1]):
    return g2_0_binary(mu, eta_s1,eta_s2, eta_i, lambdas=lambdas)#/g2_0_pnr(mu, eta_s1,eta_s2, eta_i, k=k, lambdas=lambdas)




def solveSystemEqArr_analytic(dataArr, muGuessArr, lambdas=[1],k=3,eta_s1=2*0.1828725038,eta_s2= 2*0.2145929339, eta_i = 0.312 ):
    rate = 10**6
    weights = [1,0]#np.ones(len(dataArr))/len(dataArr)
    print(weights)
    def solveSystemEq_analytic(data, muGuess):
        def myFunction(z):
            mu = z
            F=np.empty((1))
            F[0] = data[0] - rate*Ris1s2_binary_analytic(mu, eta_s1,eta_s2, eta_i, lambdas=lambdas)
            #F[0] = weights[0]*(data[0] - rate*Ris1s2_binary_analytic(mu, eta_s1,eta_s2, eta_i, lambdas=lambdas))**2+weights[1]*(data[1] - rate*Ris1s2_pnr_analytic(mu, eta_s1,eta_s2, eta_i, k=k,lambdas=lambdas))**2
            return F
        zGuess = np.array([muGuess])
        print("zGuess: ", zGuess)
        return fsolve(myFunction, zGuess)
    mu_arr=[]
    three_binary, three_pnr=dataArr
    for n in range(len(three_binary)):
        z = solveSystemEq_analytic([three_binary[n],three_pnr[n]], muGuessArr[n])
        print("z: ", z)
        print()
        mu_arr.append(z[0])
    return np.array(mu_arr)#,np.array(k_arr)

# def solveSystemEqArr_analytic(dataArr, muGuessArr, lambdas=[1]):
#     threefolds_binary, threefolds_pnr = dataArr
#     weights = np.ones(len(dataArr))/len()
#     rate = 10**6
#     eta_s1,eta_s2, eta_i=2*0.1828725038, 2*0.2145929339, 0.3123947813
#     def solveSystemEq_analytic_binary(threefolds, muGuess):
#         def myFunction(z):
#             mu = z
#             F=np.empty((1))
#             F[0] = threefolds - rate*Ris1s2_binary_analytic(mu, eta_s1,eta_s2, eta_i, lambdas=lambdas)
#             return F
#         zGuess = np.array([muGuess])
#         print("zGuess: ", zGuess)
#         return fsolve(myFunction, zGuess)
#     def solveSystemEq_analytic_pnr(threefolds, muGuess):
#         def myFunction(z):
#             mu = z
#             F=np.empty((1))
#             F[0] = threefolds - rate*Ris1s2_pnr_analytic(mu, eta_s1,eta_s2, eta_i, lambdas=lambdas)
#             return F
#         zGuess = np.array([muGuess])
#         print("zGuess: ", zGuess)
#         return fsolve(myFunction, zGuess)
#     mu_arr_binary = []
#     for n in range(len(threefolds_binary)):
#         z = solveSystemEq_analytic_binary(threefolds_binary[n], muGuessArr[n])
#         mu_arr_binary.append(z[0])
#     mu_arr_pnr = []
#     for n in range(len(threefolds_pnr)):
#         z = solveSystemEq_analytic_pnr(threefolds_pnr[n], mu_arr_binary[n])
#         mu_arr_pnr.append(z[0])
#     return np.array(mu_arr_binary),np.array(mu_arr_pnr)
