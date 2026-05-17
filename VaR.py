
"""
Created on Fri May 15 21:16:52 2026

@author: fotisgiannakis
"""

import numpy as np 
from scipy.stats import norm
from scipy.optimize import minimize 
from scipy.special import gammaln
from scipy.stats import t 
import yfinance as yf 



tickers= ['AGG','BND', 'TLT','IEF','SHY','LQD','HYG','JNK','TIP','BNDX','EMB','GOVT','VGIT','VCIT','VCSH','IGIB','USIG','FALN','USHY',
          'SCHP','VTIP','STIP','BWX','IGOV','VWOB','PCY','BOND','BSV','BIV','BLV','SPSB','SPIB','SPLB','SCHR','SCHO','SCHZ','MUB','SUB',
          'VTEB','SHM','ITM','MLN','NEAR','SHV','ICSH','JPST','GSY','FLOT','MINT','IGSB'] #MEXRI 288 GRAMMH A ERVTHMA 

#Question A - 1 Day VaR

raw = yf.download(tickers, start='2024-01-01', end='2026-01-01', auto_adjust=True)
prices = raw['Close']

prices = prices.dropna()

pr = prices.values 


log_pr = np.log(pr)


ret = log_pr[1:, :] - log_pr[:-1,:]

T, n = ret.shape 

w = np.ones(n) / n
w = w.reshape(-1,1)

ret_p = np.dot(ret,w)

ss = np.var(ret_p)
m = np.mean(ret_p)

p = 10 / 100

#VaR Normal 

def var_normal(m, ss, p):
    v = m+np.sqrt(ss) * norm.ppf(p)
    return v


v_par = var_normal(m, ss, p)
VaR_value_par = v_par * 500000000


# VaR Historical

def var_histsim(returns_hist, p):

    returns_hist = returns_hist.flatten()

    returns_hist_sort = np.sort(returns_hist)

    n = len(returns_hist)

    k = round(n * p)

    v = returns_hist_sort[k-1]

    return v


v_hist = var_histsim(ret_p , p)
VaR_value_hist = v_hist * 500000000


#stdtpdf

def stdtpdf(x, mu, sigma2, nu):

    x = np.asarray(x)
    x = x.reshape(-1, 1)


    T, K = x.shape


    if K != 1:
        raise ValueError('x must be a column vector (1D array)')
    
    if not np.isscalar(mu) and mu.shape != x.shape:
        raise ValueError('mu must be either a scalar or the same size as x') 
    

    if np.any(sigma2 <= 0):
        raise ValueError('sigma2 must contain only positive elements') 
    

    if np.isscalar(sigma2):
        sigma2 = sigma2 * np.ones((T, K))
    elif sigma2.shape[0] != T or sigma2.shape[1] != 1:
        raise ValueError('sigma2 must be a scalar or a vector with the same dimensions as X') 

    

    if not np.isscalar(nu) or nu <= 2:
        raise ValueError('nu must be a scalar greater than 2')
    

    x = x - mu


    constant = np.exp(gammaln(0.5 * (nu + 1)) - gammaln(0.5 * nu))

    y =  constant / np.sqrt(np.pi * (nu - 2) * sigma2) * (1+ (x ** 2) / (sigma2 * (nu - 2))) ** (-(nu + 1) / 2)

    y = np.maximum(y, 1e-10)


    return y


# iscompatible

def iscompatible(narg, *args):
    errortext = ''
    varargout = [None] * narg


    if len(args) < narg or any(arg is None or (hasattr(arg, '__len__') and len(arg) == 0) for arg in args):
        sizeOut = []
        return True, 'Too few parameters. All inputs must be nonempty.', sizeOut, varargout

    params = args[:narg]
    param_len = [np.size(p) for p in params]


    if all (l == 1 for l in param_len):
        param_size = (1,1)

    elif sum (l>1 for l in param_len) == 1:
        idx = next(i for i, l in enumerate(param_len) if l>1) 
        param = params[idx]
        param_size = np.shape(param)


    else:

        param_non_scalar = [i for i, l in enumerate(param_len) if l > 1]  
        params_non_scalar = [params[i] for i in param_non_scalar]
        
        for i in range(len(params_non_scalar) - 1):
            if np.shape(params_non_scalar[i]) != np.shape(params_non_scalar[i+1]):
                sizeOut =[]
                error = 1
                errortext = 'Parameter size mismatch. Must be either scalar or of a common size.'
                return error, errortext, sizeOut, varargout
        param_size = np.shape(params_non_scalar[0])

            
    if len(args) > narg:
        sizeOut = args[narg:]

        if len(sizeOut) == 1:
            sizeOut = sizeOut[0]

            if len(np.shape(sizeOut)) == 2 and np.size(sizeOut) == np.prod(np.shape(sizeOut)):
                pass
            else:
                sizeOut = []
                error = 1
                errortext = 'Requested output size cannot be parsed. Should be a vector or series of scalars.'
                return error, errortext, sizeOut, varargout

        else:
            if np.prod(param_size) != 1 and sizeOut:
                if np.array_equal(param_size, sizeOut):
                    error = 0
                else:
                    error = 1
                    sizeOut = []
                    errortext = 'Requested output size and parameters are not of compatible sizes.'
                    return error, errortext, sizeOut, varargout

            elif not sizeOut:
                sizeOut = param_size
                error = 0
            else:
                error = 0

   
    if len(args) > 3:
        for i in range(narg):
            if len(args[i]) == 1:
                varargout[i] = np.tile(args[i], sizeOut)
            else:
                varargout[i] = args[i]

    return error, errortext, sizeOut, varargout





 #STDTINV

def stdtinv(p, v):
    if p is None or v is None:
        raise ValueError(f"Invalid value for v: {v}. v must be greater than 2.")

    x = t.ppf(p, v)

    stdev = np.sqrt(v / (v-2))

    stdev = np.where(v <= 2, np.nan, stdev)

    x = x / stdev

    return x 


# Log- Likelihood

def loglikel_st(theta, r):
    mu = theta[0]
    ss = theta[1]
    v = theta[2]

    T = len(r)
    log_lik = np.zeros(T)


    for i in range(T):

        log_lik[i] = np.log(stdtpdf(np.array([[r[i]]]), mu, ss, v)[0, 0])

    total_LL = -np.sum(log_lik)
    return total_LL
    
theta0 = np.array([0, 1, 3])


# Bounds 

bounds = [(-100, 100), (1e-10, 1000), (2.01, 300)]



options = {'disp': True}

res = minimize(lambda theta: loglikel_st(theta, ret_p), theta0, bounds = bounds, options = options)


thetaStar_2 = res.x 
mu_st = thetaStar_2[0]
ss_st = thetaStar_2[1]
v_st = thetaStar_2[2]



p = 0.10 

v_st = mu_st +np.sqrt(ss_st) * stdtinv(p,v_st)

VaR_st = v_st * 500000000


#Question B 

# Backtesting

def backtest_param(prr, k, p):
    Tp, n = prr.shape
    g_par = np.zeros(k)
    g_hist = np.zeros(k)
    g_st = np.zeros(k)
    res = np.zeros((k, 4))

    for i in range(k):
      t_idx = Tp - i - 1
      pr = prr[:t_idx, :]

      if t_idx +1 < Tp:
        prf = prr[t_idx:t_idx + 2, :]
      else:
        continue

      log_pr = np.log(pr)
      log_prf = np. log(prf)

      ret = log_pr[1:] - log_pr[:-1]
      retf = log_prf[1] - log_prf[0]

      w = np.ones(n) / n
      ret_p = ret @ w
      ret_pf = np.dot(retf , w)

      ss = np.var(ret_p)
      m = np.mean(ret_p)

      theta0 = np.array([0 , 1, 3])
      bounds = [(-100 , 100), (1e-10, 1000), (2.01, 300)]

      result = minimize(lambda theta: loglikel_st(theta, ret_p),theta0, bounds = bounds, options = options)

      mu_st, ss_st, v_st_param = result.x

      v_par = var_normal(m, ss, p)
      v_hist = var_histsim(ret_p, p)
      v_st = mu_st + np.sqrt(ss_st) * stdtinv(p, v_st_param)

      if ret_pf < v_par:
        g_par[i] = 1
      else:
        g_par[i] = 0

    
      if ret_pf < v_hist:
        g_hist[i] = 1
      else:
        g_hist[i] = 0
    
    
      if ret_pf < v_st:
        g_st[i] = 1
      else:
        g_st[i] = 0



      res[i, 0] = np.array(v_hist).item()  
      res[i, 1] = np.array(v_par).item()   
      res[i, 2] = np.array(v_st).item()  
      res[i, 3] = np.array(ret_pf).item()         


    ch_parm = np.sum(g_par) / k
    ch_hist = np.sum(g_hist) / k
    ch_st = np.sum(g_st) / k

    return ch_parm, ch_hist, ch_st, res


k = 300

ch_parm, ch_hist, ch_st, res = backtest_param(pr, k, p)



# Question C — 5-Day VaR


h = 5

#1. Historical Simulation 5-day VaR
ret_p_5day = np.array([
    np.sum(ret_p[i:i+h]) 
    for i in range(len(ret_p) - h + 1)
])

v_hist_5 = np.percentile(ret_p_5day, p * 100)
VaR_hist_5 = v_hist_5 * 500000000

#2. Parametric Normal 5-day VaR 
m_5 = m * h
ss_5 = ss * h

v_par_5 = var_normal(m_5, ss_5, p)
VaR_par_5 = v_par_5 * 500000000

#3. Parametric Student-t 5-day VaR
mu_st_5 = mu_st * h
ss_st_5 = ss_st * h

v_st_5 = mu_st_5 + np.sqrt(ss_st_5) * stdtinv(p, thetaStar_2[2])
VaR_st_5 = v_st_5 * 500000000



    
                                                                                                 

     

         









































































