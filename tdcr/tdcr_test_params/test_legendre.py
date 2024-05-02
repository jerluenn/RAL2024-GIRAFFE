import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

x_data = np.linspace(0, 0.5, 10)
y_data = np.array([1.        , 0.34725257, 0.34664312, 0.34619106, 0.34586532,
       0.34563938, 0.34549123, 0.34540241, 0.34535681, 0.34533972])

def polynomial_regression(x_data, y_data, degree):
    # Construct the Vandemonde matrix
    vander_matrix = np.vander(x_data, degree + 1, increasing=True)
    
    # Perform polynomial regression
    coefficients = np.linalg.lstsq(vander_matrix, y_data, rcond=None)[0]
    
    return coefficients

def evaluate_polynomial(coefficients, x):
    return np.polyval(coefficients[::-1], x)

def model(y, t):
    # Define the differential equation
    dydt = 0 

    for i in range(1, coefficients.size):

        dydt += i*coefficients[i]*t**(i-1)

    return dydt

# Example usage:
# x_data = np.array([0, 1, 2, 3, 4, 5])  # Your x values
# y_data = np.array([2.1, 7.7, 13.6, 27.2, 40.9, 61.1])  # Your y values
degree = 10  # Degree of the polynomial

global coefficients

coefficients = polynomial_regression(x_data, y_data, degree)
print("Coefficients:", coefficients)

# Initial condition
y0 = 1

# Time points to solve the ODE for
t = np.linspace(0, 0.5, 10)

# Solve the ODE
y = odeint(model, y0, t)

# Plotting
x_range = np.linspace(min(x_data), max(x_data), 100)
y_fitted = evaluate_polynomial(coefficients, x_range)

plt.figure(figsize=(8, 6))
plt.scatter(x_data, y_data, label='Data Points', color='blue')
plt.plot(x_range, y_fitted, label='Fitted Polynomial', color='red')
plt.plot(x_data, y, label='ODEint', color='green')
plt.title('Polynomial Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
