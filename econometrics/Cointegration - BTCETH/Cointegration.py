    
# -*- coding: utf-8 -*-
from statsmodels.tsa.tsatools import (lagmat, add_trend)
import numpy as np
import pandas as pd

def GetOLS(Y,X):
    nobs = X.shape[1]
    rank = X.shape[0]
    #noofVariables = X.shape[0]
    covariance = np.linalg.inv(np.dot(X, X.T))  # [(ZZ')^-1] variance-covariance factor
    beta_hat = np.dot(np.dot(Y, X.T,), covariance) # beta_hat YZ'(ZZ')^-1
    if beta_hat.ndim < 2:
        beta_hat = beta_hat[None,:]
        
    resid_hat = Y - np.dot(beta_hat, X) # resid_hat = Y - beta_hat*Z
    df_resid = np.float(nobs - rank)
    rri = np.dot(resid_hat, resid_hat.T)  # resid_hat*resid_hat' 
    
    sigma_hat = rri / nobs #  'sigma_hat' -Estimator of the residual covariance martrix with T = Nobs
    ols_scale = rri / df_resid # OLS estimator for cov matrix
    
    cov_params = np.kron(covariance, ols_scale)  # covariance matrix of parameters
    bvar = np.diag(cov_params)  # variances - diagonal of covariance matrix
    stderr = np.sqrt(bvar)  # standard error
    stderr = stderr.reshape((beta_hat.shape[0], beta_hat.shape[1]), order='C')
    
    tvalues = beta_hat / stderr  # t-statistic for a given parameter estimate
    nobs2 = nobs / 2.0
    llf = -nobs2 * np.log(2 * np.pi) - nobs2 * np.log(sigma_hat) - nobs2  # log-likelihood function
    df_model = rank  # degrees of freedom of model
    eigenvalues = np.roots(np.r_[1,-beta_hat[0]])  # eigen values
    roots = eigenvalues ** -1  #roots
    
    is_Stable = np.all(np.abs(roots) > 1)
    resultOLS = {'X': X,
    'Y': Y,
    'beta_hat': beta_hat,
    'resid_hat': resid_hat,
    'nobs': nobs,
    'df_resid': df_resid,
    'rri': rri,
    'sigma_hat': sigma_hat,
    'ols_scale': ols_scale,
    'cov_params': cov_params,
    'stderr': stderr,
    'tvalues': tvalues,
    'llf': llf,
    'df_model': df_model,
    'roots': roots,
    'is_Stable': is_Stable
    }
    return resultOLS

def GetADFuller(Y, maxlags=None, regression='c'):
    #Y = np.asarray(Y)
    Y = Y.T
    dy = np.diff(Y)
    if dy.ndim == 1:
        dy = dy[:, None]
    ydall = lagmat(dy, maxlags, trim='both', original='in')
    nobs = ydall.shape[0] 
    ydall[:, 0] = Y[-nobs - 1:-1] 
    dYshort = dy[-nobs:] 
    
    if regression != 'nc':
        Z = add_trend(ydall[:, :maxlags + 1], regression) 
    else:
        Z = ydall[:, :maxlags + 1]
    
    resultADFuller = GetOLS(Y=dYshort.T, X=Z.T)  
    
    K_dash = 2 * (2*maxlags + 1)
    AIC = np.log(np.absolute(np.linalg.det(resultADFuller['sigma_hat']))) + 2.0 * K_dash / (resultADFuller['nobs']) # log(sigma_hat) + 2*K_dash/T        
    BIC = np.log(np.absolute(np.linalg.det(resultADFuller['sigma_hat']))) + (K_dash/ resultADFuller['nobs']) * np.log(resultADFuller['nobs']) # log(sigma_hat) + K_dash/T*log(T)
    resultADFuller['AIC'] = AIC
    resultADFuller['BIC'] = BIC
    resultADFuller['adfstat'] = resultADFuller['tvalues'][0,0]
    resultADFuller['maxlag'] = maxlags
    
    return resultADFuller

def GetVectorAR(Y, maxlags=None,  trend=None):
    nobs = Y.shape[1]
    Yshort = Y[:,maxlags:]
    Z =  np.ones(nobs-maxlags)
    if maxlags == 0:
        Z = np.ones(nobs-maxlags)[None,:]
    else:
        for j in range(1,maxlags+1):
            Z = np.vstack((Z, Y[:,maxlags-j:-j]))
    if trend is not None:
        Z = add_trend(Z.T, prepend=True, trend=trend)  # prepends puts trend column at the beginning
        Z = Z.T

    resultVectorAR = GetOLS(Y=Yshort, X=Z)
    resultVectorAR['maxlags'] = maxlags
    K_dash = 2 * (2* maxlags + 1)
    AIC = np.log(np.absolute(np.linalg.det(resultVectorAR['sigma_hat']))) + 2.0 * K_dash / (resultVectorAR['nobs']) # log(sigma_hat) + 2*K_dash/T        
    BIC = np.log(np.absolute(np.linalg.det(resultVectorAR['sigma_hat']))) + (K_dash/ resultVectorAR['nobs']) * np.log(resultVectorAR['nobs']) # log(sigma_hat) + K_dash/T*log(T)
    resultVectorAR['AIC'] = AIC
    resultVectorAR['BIC'] = BIC
    
    return resultVectorAR
		
def GetOptimalLag(Y,maxlags, modelType='VectorAR'):
    result={}
    for nlag in range(0, maxlags+1):
        if modelType == 'VectorAR':
            result[nlag] = GetVectorAR(Y, maxlags=nlag)
        elif modelType == 'ADFuller':
            result[nlag] = GetADFuller(Y=Y, maxlags=nlag, regression='constant')
    aicbest, bestlagaic = min((v['AIC'], k) for k, v in result.items())
    bicbest, bestlagbic = min((v['BIC'], k) for k, v in result.items())
    results = {'aicbest': aicbest,
    'bestlagaic': bestlagaic,
    'bicbest': bicbest,
    'bestlagbic': bestlagbic
    }
    return results
#end GetOptimalLag
def IsStable(roots):
    return np.all(np.abs(roots) > 1)


def GetZScore(series, mean=None, sigma=None):
    if (mean != None and sigma != None):
        return (series - mean) / sigma
    else:
        return (series - series.mean()) / np.std(series)

def Get_Pnl_DF(spread, mean, sigma):
    """
    Note the input spread must be zscore-normalised
    """
    spread_norm = (spread - mean) / sigma  # normalise as z-score
    df_pnl_is = pd.DataFrame(index=spread.index)
    df_pnl_is['e_t_hat'] = spread
    df_pnl_is['e_t_hat_norm'] = spread_norm
    # df_pnl_is['diff'] = df_pnl_is['e_t_hat'].diff()
    df_pnl_is['pos'] = np.nan
    # Go long the spread when it is below -1 as expectation is it will rise
    df_pnl_is.loc[df_pnl_is['e_t_hat_norm'] <= -1.0, 'pos'] = 1
    # Go short the spread when it is above +1 as expectation is it will fall
    df_pnl_is.loc[df_pnl_is['e_t_hat_norm'] >= 1.0, 'pos'] = -1
    # Exit positions when close to zero
    df_pnl_is.loc[(df_pnl_is['e_t_hat_norm'] < 0.1) & (df_pnl_is['e_t_hat_norm'] > -0.1), 'pos'] = 0
    # # forward fill NaN's with previous value
    df_pnl_is['pos'] = df_pnl_is['pos'].fillna(method='pad')

    # Returns must be calculated in unnormalised spread
    df_pnl_is['chg'] = df_pnl_is['e_t_hat'].diff().shift(
        -1)  # adopting Boris convention with shift(-1) (must shift after taking diff)
    # PnL
    df_pnl_is['pnl'] = df_pnl_is['pos'] * df_pnl_is['chg']
    df_pnl_is['pnl_cum'] = df_pnl_is['pnl'].cumsum()

    return df_pnl_is