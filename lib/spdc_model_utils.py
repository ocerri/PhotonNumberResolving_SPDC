import numpy as np

def countProbaSignal(mu, eta, lambdas=np.ones(1)):
    # eta * mu * lambda
    eml = np.atleast_2d(eta*mu).T * np.atleast_1d(lambdas)
    out = 2./ ( 2 + eml )
    out = 1 - np.product(out, axis=1)

    if isinstance(mu, float):
        return out[0]
    else:
        return out



def countProbaIdler(mu, eta, k=0, lambdas=np.ones(1)):
    # eta * mu * lambda
    eml = np.atleast_2d(eta*mu).T * np.atleast_1d(lambdas)
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



def countProbaiS(mu, eta_i, eta_s, k=0, lambdas=np.ones(1)):
    kk = 2**k

    ei_m_l = np.atleast_2d(eta_i*mu).T * np.atleast_1d(lambdas)

    term1 = kk / (kk + (kk - 1)*ei_m_l)
    term1 = np.product(term1, axis=1)

    aux = mu*eta_s*(kk - (kk - 1)*eta_i)
    aux = np.atleast_2d(aux).T * np.atleast_1d(lambdas)
    term2 = 2*kk / (aux + 2*(kk + (kk - 1)*ei_m_l))
    term2 = np.product(term2, axis=1)

    term3 = np.product(1./ (1. + ei_m_l), axis=1)

    aux = np.atleast_2d(eta_s*mu*(1-eta_i)).T * np.atleast_1d(lambdas)
    term4 = 2./ (2. + 2*ei_m_l + aux)
    term4 = np.product(term4, axis=1)

    out = kk*(term1 - term2 - term3 + term4)

    if isinstance(mu, float):
        return out[0]
    else:
        return out



def countProbai12(mu, eta_i, eta_1, eta_2, k=0, lambdas=np.ones(1)):
    kk = 2**k

    ei_m_l = np.atleast_2d(eta_i*mu).T * np.atleast_1d(lambdas)

    term1 = kk / (kk + (kk - 1)*ei_m_l)
    term1 = np.product(term1, axis=1)

    aux = mu*eta_1*(kk - (kk - 1)*eta_i)
    aux = np.atleast_2d(aux).T * np.atleast_1d(lambdas)
    term2 = 2*kk / (aux + 2*(kk + (kk - 1)*ei_m_l))
    term2 = np.product(term2, axis=1)

    aux = mu*eta_2*(kk - (kk - 1)*eta_i)
    aux = np.atleast_2d(aux).T * np.atleast_1d(lambdas)
    term3 = 2*kk / (aux + 2*(kk + (kk - 1)*ei_m_l))
    term3 = np.product(term3, axis=1)

    aux = mu*(eta_1+eta_2)*(kk - (kk - 1)*eta_i)
    aux = np.atleast_2d(aux).T * np.atleast_1d(lambdas)
    term4 = 2*kk / (aux + 2*(kk + (kk - 1)*ei_m_l))
    term4 = np.product(term4, axis=1)

    term5 = np.product(1./ (1. + ei_m_l), axis=1)

    aux = np.atleast_2d(eta_1*mu*(1-eta_i)).T * np.atleast_1d(lambdas)
    term6 = 2./ (2. + 2*ei_m_l + aux)
    term6 = np.product(term6, axis=1)

    aux = np.atleast_2d(eta_2*mu*(1-eta_i)).T * np.atleast_1d(lambdas)
    term7 = 2./ (2. + 2*ei_m_l + aux)
    term7 = np.product(term7, axis=1)

    aux = np.atleast_2d((eta_1 + eta_2)*mu*(1-eta_i)).T * np.atleast_1d(lambdas)
    term8 = 2./ (2. + 2*ei_m_l + aux)
    term8 = np.product(term8, axis=1)

    out = 2**k*(term1 - term2 - term3 + term4 - term5 + term6 + term7 - term8)
    if isinstance(mu, float):
        return out[0]
    else:
        return out



def g2_0(mu, eta_i, eta_1, eta_2, k=0, lambdas=np.ones(1)):
    out = countProbai12(mu, eta_i, eta_1, eta_2, k, lambdas) * countProbaIdler(mu, eta, k, lambdas)
    out /= countProbaiS(mu, eta_i, eta_1, k, lambdas) * countProbaiS(mu, eta_i, eta_2, k, lambdas)
    return out
