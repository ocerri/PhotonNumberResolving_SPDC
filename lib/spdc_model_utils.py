import numpy as np

def countProbaSignal(mu, eta, lambdas=np.ones(1)):
    # eta * mu * lambda
    eml = eta * np.atleast_2d(mu).T * np.atleast_1d(lambdas)
    out = 2./ ( 2 + eml )
    out = 1 - np.product(out, axis=1)

    if isinstance(mu, float):
        return out[0]
    else:
        return out

def countProbaIdler(mu, eta, k=0, lambdas=np.ones(1)):
    # eta * mu * lambda
    eml = eta * np.atleast_2d(mu).T * np.atleast_1d(lambdas)
    kk = 2**k
    term1 = np.product(kk / ( kk + (kk - 1.)*eml ), axis=1)
    term2 = np.product(1./ ( 1. + eml ), axis=1)
    out = kk*(term1 - term2)

    if isinstance(mu, float):
        return out[0]
    else:
        return out


def countProbaS1S2(mu, eta1, eta2, lambdas=np.ones(1)):
    t_s1  = 1. - countProbaSignal(mu, eta1, lambdas)
    t_s2  = 1. - countProbaSignal(mu, eta2, lambdas)
    t_s12 = 1. - countProbaSignal(mu, eta1+eta2, lambdas)
    return 1 - t_s1 - t_s2 + t_s12
