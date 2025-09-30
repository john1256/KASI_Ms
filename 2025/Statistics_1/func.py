import numpy as np
from scipy.special import gamma
from scipy.integrate import quad

def mean(data):
    return np.sum(data)/len(data)
def geom_mean(data):
    if np.any(data<=0):
        return np.nan
    else:
        return np.exp(np.sum(np.log(data))/len(data))
def median(data):
    sort = np.sort(data)
    n = len(data)
    if n%2==0:
        return (sort[n//2-1]+sort[n//2])/2
    else:
        return sort[n//2]
def variance(data):
    return mean(data**2)-mean(data)**2
def std(data):
    return np.sqrt(variance(data))
def skewness(data):
    x3 = mean(data**3)
    x2 = mean(data**2)
    x = mean(data)
    sig = std(data)
    return 1/(sig**3)*(x3 - 3*x*x2 + 2*x**3)
def kurtosis(data):
    mu = mean(data)
    return mean((data - mu)**4) / (variance(data)**2) - 3
def weighted_mean(data, stds):
    weights = 1/stds**2
    return np.sum(data*weights)/np.sum(weights)
def weighted_std(stds):
    weights = 1/stds**2
    return np.sqrt(1/np.sum(weights))

def Chisq_pdf(chisq, n):
    chi = np.sqrt(chisq)
    prob = 2**(-n/2)/gamma(n/2)*chi**(n-2)*np.exp(-chisq/2)
    return prob
def Chisq_integral(chisq, n):
    result, _ = quad(Chisq_pdf, chisq, np.inf, args = (n,))
    return result