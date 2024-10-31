# TREEMATE/econ.py
import numpy as np
import statsmodels.api as sm
from .mathematics import functionvalue, integrate_poly

def market_equilibrium(supp, demd, maxdif=1.,maxiter=1e3):
    """
    Finds the market equilibrium points where the supply and demand polynomials are equal.
# 
    Parameters:
    supp (list or array): Coefficients of the supply polynomial.
    demd (list or array): Coefficients of the demand polynomial.
    maxdif (float): Maximum prices difference on equilibrium.
    maxiter (int): Maximum number of iterations

    Returns:
    tuple: A tuple containing:
        - quantity in equilibrium
        - price in equilibrium
        - final iteration step
        - final difference in prices between demand and supply in returned quantity
        - number of iterations
    """

    delta = 1.
    q = [5.0]
    i=0
    sup = functionvalue(supp, q, "poly")
    dem = functionvalue(demd, q, "poly")
    
    while (np.abs(sup[0]-dem[0]) > maxdif) and (i<maxiter):
        i += 1
        q.append(q[0]+delta)
        sup = functionvalue(supp, q, "poly")
        dem = functionvalue(demd, q, "poly")
        dy = np.abs(sup[0]-dem[0])
        dy2 = np.abs(sup[1]-dem[1])
        if (dy2 < dy):
            q=[q[0]+delta]
        else:
            q=[q[0]-delta]
        if (sup[0]-dem[0])*(sup[1]-dem[1]) < 0:
            delta = delta/2

    return(q[0],(sup[0]+dem[0])*0.5,delta,np.abs(sup[0]-dem[0]),i)    

def surplus(a,equilibrium_q, equilibrium_p=0):
    """
    Computes the surplus by finding the difference between the area under the
    polynomial curve and the area of the rectangle formed by the equilibrium price
    and quantity.
    
    Parameters:
    a (list or array-like): Coefficients of the polynomial in increasing powers.
    equilibrium_q (float): Market equilibrium quantity.
    
    Returns:
    float: Surplus calculated as the absolute difference between the integral and the rectangle area.
    """
    if equilibrium_p == 0:
        equilibrium_p = functionvalue(a, [equilibrium_q])[0]
    rectangle = equilibrium_p*equilibrium_q
    integral = integrate_poly(a, 0, equilibrium_q)
    return np.abs(integral - rectangle)


def estimate_price_elasticity_of_supply(Q, P):
    """
    Estimates the price elasticity of supply based on observed quantities and prices 
    by fitting a power function model to the data.
    
    Parameters:
    Q (array-like): Quantities supplied, representing the levels of supply.
    P (array-like): Prices associated with the corresponding supply quantities.
    
    Returns:
    k (float): Coefficient that scales the supply model function.
    pp (float): Intercept term representing base price in the model.
    epsilon (float): Estimated elasticity of supply, measuring the responsiveness of 
                     quantity supplied to changes in price.
    covariance (ndarray): Covariance matrix of the estimated parameters, indicating 
                          parameter uncertainties.
    """
    from scipy.optimize import curve_fit

    # Define the model function for curve fitting
    def model(q, k, pp, epsilon):
        return k * q**(1/epsilon) + pp

    initial_guess = [15./100000., 30, 1]  # Initial guesses for k, pp, and epsilon
    params, covariance = curve_fit(model, Q, P, p0=initial_guess, maxfev=2000)
    k, pp, epsilon = params    
    return k, pp, epsilon, covariance


def price_estimation_on_es_value(k, p0, q, epsilon):
    """
    Estimates the price based on a given quantity using the supply model parameters.

    Parameters:
    k (float): Scale factor in the supply model.
    p0 (float): Base price intercept in the model.
    q (float or array-like): Quantity for which the price is to be estimated.
    epsilon (float): Elasticity of supply, indicating how responsive the supply is to price changes.
    
    Returns:
    float or ndarray: Estimated price for the given quantity.
    """
    return k * np.power(q, 1./epsilon) + p0


def producers_surplus(q, k, p0, epsilon):
    """
    Calculates the producer's surplus, representing the benefit producers receive 
    when they sell at the market price.

    Parameters:
    q (float or array-like): Quantity supplied.
    k (float): Scale factor in the supply model.
    p0 (float): Base price intercept in the model.
    epsilon (float): Elasticity of supply.

    Returns:
    float or ndarray: Producer's surplus for the given quantity.
    """
    p = price_estimation_on_es_value(k, p0, q, epsilon)
    return p * q - p0 * q - k * (epsilon / (epsilon + 1) * np.power(q, 1 + 1. / epsilon))


