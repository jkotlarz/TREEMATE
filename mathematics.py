# TREEMATE/mathematics.py
import numpy as np
import statsmodels.api as sm

def functionvalue(a, x, functiontype="poly"):
    """
    Compute function values at given arguments using coefficients.

    Args:
    - a (list): List of coefficients representing the function.
    - x (list): List of values at which the function should be evaluated.
    - functiontype (str, optional): Type of function (default is "poly").

    Returns:
    - list: List of computed function values corresponding to each x.

    Description:
    This function computes the function values based on the coefficients `a`.
    It assumes `a` represents coefficients of a polynomial unless specified otherwise.
    For each `xi` in `x`, it calculates `yi` using the formula:
    yi = sum(coef * xi**power for power, coef in enumerate(a))
    """
    y = []
    for k in range(len(x)):
        xi = x[k]
        yi = a[0]
        xv=1.0
        for i in range(1,len(a)):
            xv = xv*xi
            yi = yi + xv*a[i] 
        y.append(yi)
    return y


    
def integrate_poly(a, x0, x1):
    """
    Integrate a polynomial defined by its coefficients over the interval [x0, x1].

    Args:
    - a (list): Coefficients of the polynomial in ascending order of degree.
    - x0 (float): Lower bound of integration.
    - x1 (float): Upper bound of integration.

    Returns:
    - float: Definite integral of the polynomial over [x0, x1].

    Description:
    This function computes the definite integral of a polynomial defined by its coefficients `a`
    over the interval [x0, x1]. The coefficients `a` are assumed to be in ascending order of degree.
    It first adjusts the coefficients to account for integration, computes the values of the integrated
    polynomial at `x0` and `x1` using `functionvalue`, and returns the difference between these two values.
    """
    b=[0]
    for i in range(len(a)):
        b.append(a[i])
#    b = a.insert(0, 0)  # Insert a_0 = 0 to handle integration constant
    for i in range(1, len(b)):
        b[i] = b[i] / i  # Divide each coefficient by its degree to integrate

    y = functionvalue(b, [x0, x1])  # Compute function values at x0 and x1
    return y[1] - y[0]  # Return the difference to get the definite integral

    

def estimate_poly(x, y, n):
    """
    Estimate polynomial coefficients and Mean Squared Error (MSE) using least squares fit.

    Args:
    - x (list or numpy array): Array of x-coordinates.
    - y (list or numpy array): Array of y-coordinates.
    - n (int): Degree of the polynomial to be fitted.

    Returns:
    - tuple: Coefficients of the polynomial of degree n and the Mean Squared Error (MSE).

    Description:
    This function computes the coefficients of the polynomial of degree n
    that best fits the given points (x, y) using the method of least squares.
    It also calculates the Mean Squared Error (MSE) to evaluate the quality of the fit.
    """
    # Ensure x and y are numpy arrays for numerical operations
    x = np.array(x)
    y = np.array(y)

    # Create the Vandermonde matrix
    A = np.vander(x, n + 1, increasing=True)

    # Solve the least squares problem
    coefficients = np.linalg.lstsq(A, y, rcond=None)[0]

    # Calculate predicted y values
    y_pred = np.dot(A, coefficients)

    # Calculate MSE
    mse = np.mean((y - y_pred)**2)

    return coefficients, mse

def array_difference(a, b):
    """
    Compute the element-wise difference between two arrays.
    
    Parameters:
    a (list): First input list of numbers.
    b (list): Second input list of numbers.
    
    Returns:
    list: A list containing the differences between corresponding elements of `a` and `b` if they have the same length.
    bool: Returns False if the lengths of `a` and `b` are not equal.
    """
    if len(a) == len(b):
        c = []
        for i in range(len(a)):
            c.append(a[i] - b[i])
        return c
    else:
        return False

def pearson_correlation(x, y):
    """
    Calculate the Pearson correlation coefficient between two lists of numbers.
    
    Parameters:
    x (list): First list of numbers.
    y (list): Second list of numbers.
    
    Returns:
    float: Pearson correlation coefficient between `x` and `y`.
    """
    # Check if lengths of x and y are the same
    if len(x) != len(y):
        raise ValueError("Lists must have the same length.")
    
    # Calculate mean of x and y
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Calculate numerator and denominators for Pearson correlation coefficient
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator_x = np.sqrt(sum((xi - mean_x)**2 for xi in x))
    denominator_y = np.sqrt(sum((yi - mean_y)**2 for yi in y))
    
    # Calculate Pearson correlation coefficient
    correlation_coefficient = numerator / (denominator_x * denominator_y)
        
    return correlation_coefficient

def smooth_array(a, window_size=3):
    """
    Smooths the array `a` using a moving average filter with the specified `window_size`.
    The output array will have the same length as the input array.
    
    Parameters:
    - a: The input array (list or numpy array) to smooth
    - window_size: The size of the moving average window, default is 3
    
    Returns:
    - A smoothed array with the same length as the input array
    """
    a = np.array(a)  # Convert to numpy array if it is not already
    if window_size < 1:
        raise ValueError("Window size must be greater than or equal to 1")
    
    # Apply convolution with 'same' mode to maintain the same length
    smooth_a = np.convolve(a, np.ones(window_size)/window_size, mode='same')
    
    return smooth_a

def consecutive_differences(v):
    """
    Calculates the difference between consecutive elements in the array `v`.
    
    Parameters:
    - v: The input array (list or numpy array)
    
    Returns:
    - An array of differences between consecutive elements of `v`
    """
    v = np.array(v)  # Convert to numpy array if it is not already
    
    # Calculate the differences between consecutive elements
    differences = np.diff(v)
    differences = np.insert(differences, 0, v[0])

    
    return differences

def construct_v(dv):
    """
    Constructs the array v from the array dv such that:
    - v[0] = dv[0]
    - v[n+1] = v[n] + dv[n] for n >= 0
    
    Parameters:
    - dv: The input array (list or numpy array) of differences
    
    Returns:
    - The constructed array v
    """
    dv = np.array(dv)  # Convert to numpy array if it is not already
    
    # Initialize v with the same length as dv plus one for the starting value
    v = np.zeros(len(dv) + 1)
    
    # Set the first element
    v[0] = dv[0]
    
    # Compute the rest of the elements
    for n in range(1, len(dv) + 1):
        v[n] = v[n - 1] + dv[n - 1]
    
    return v