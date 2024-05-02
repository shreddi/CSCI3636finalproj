import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import pandas as pd

verbose = False

# Load the dataset
data = pd.read_csv('biweekly_cases.csv')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

us_cases = data['United States']

if(verbose):
    us_cases.plot(title='COVID-19 Cases in the United States')
    plt.xlabel('Date')
    plt.ylabel('Number of Cases')
    plt.show()

wave_start = '2020-03-01'
wave_end = '2021-07-01'

# Filter data to focus on the specific wave
wave_data = us_cases.loc[wave_start:wave_end]

if(verbose):
    # Plot to verify the wave
    wave_data.plot(title='Selected COVID-19 Wave in the United States')
    plt.xlabel('Date')
    plt.ylabel('Number of Cases')
    plt.show()

# Assuming a total population size, adjust accordingly
N = 331000000  # Approximate US population
I0 = wave_data.iloc[0]
S0 = N - I0
R0 = 0  # Minimal recovered at the start of the wave

initial_conditions = [S0/N, I0/N, R0/N]  # Normalized

def SIR(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Time array for the data points
t = np.arange(len(wave_data))

# Function to integrate the differential equations
def fit_odeint(x, beta, gamma):
    return odeint(SIR, initial_conditions, t, args=(beta, gamma))[:,1]

# Objective function
def objective(params):
    beta, gamma = params
    sim_I = fit_odeint(t, beta, gamma)
    return np.mean((sim_I - (wave_data.values/N))**2)  # Mean squared error


# Initial guess for the parameters
params_guess = [0.3, 0.1]

# Bounds for the parameters
bounds = [(0.0001, 1), (0.0001, 1)]

# Perform the minimization
result = minimize(objective, params_guess, bounds=bounds, method='L-BFGS-B')
beta_opt, gamma_opt = result.x
print(beta_opt, gamma_opt)

# Get the optimized model output
optimized_I = fit_odeint(t, beta_opt, gamma_opt)
optimized_I = fit_odeint(t, beta_opt, gamma_opt)

plt.figure(figsize=(10, 6))
plt.plot(wave_data.index, wave_data.values, 'r', label='Actual Infections')
plt.plot(wave_data.index, optimized_I*N, 'b', label='Fitted Model')
plt.xlabel('Date')
plt.ylabel('Number of Cases')
plt.legend()
plt.title('Fit of SIR Model to first COVID-19 Wave')
plt.show()