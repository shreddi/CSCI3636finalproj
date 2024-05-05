import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load data for total cases, new cases, and vaccines administered in the US
total_cases_data = pd.read_csv('total_cases.csv')
total_cases_data['date'] = pd.to_datetime(total_cases_data['date'])
total_cases_data.set_index('date', inplace=True)
us_total_cases = total_cases_data['United States']

recovery_delay = pd.DateOffset(days=14)
us_recovered = total_cases_data['United States'].shift(-recovery_delay.days)

new_cases_data = pd.read_csv('biweekly_cases.csv')
new_cases_data['date'] = pd.to_datetime(new_cases_data['date'])
new_cases_data.set_index('date', inplace=True)
us_new_cases = new_cases_data['United States']

vax_data = pd.read_csv('USvaccinations.csv')
vax_data['date'] = pd.to_datetime(vax_data['date'])
vax_data.set_index('date', inplace=True)

wave2_start = '2021-07-01'
wave2_end = '2023-02-01'

# Filter data for the first wave period
us_total_cases_wave2 = us_total_cases[wave2_start:wave2_end]
us_new_cases_wave2 = us_new_cases[wave2_start:wave2_end]
us_recovered_wave2 = us_recovered[wave2_start:wave2_end]
us_vax_wave2 = vax_data.loc[wave2_start:wave2_end, 'people_fully_vaccinated']


#define parameters for SIR Model
N = 331000000  # Approximate US population in 2020
I0 = us_new_cases_wave2.iloc[0]  # Initial number of infected individuals
R0 = 0  # Initial number of recovered individuals
S0 = N - I0 - R0  # Initial susceptible population

# Display initial conditions
print(f"Initial conditions - S0: {S0}, I0: {I0}, R0: {R0}")

def sir_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def fit_odeint(t, beta, gamma):
    return odeint(sir_model, (S0, I0, R0), t, args=(N, beta, gamma))[:,1]

# Total days for simulation
t = np.arange(len(us_total_cases_wave2))

# Least squares parameter estimation
result = minimize(lambda x: np.mean((fit_odeint(t, *x) - us_new_cases_wave2.values)**2),
                  [0.1, 0.1], method='L-BFGS-B', bounds=[(0.0001, 1), (0.0001, 1)])
beta_opt, gamma_opt = result.x

print(f"Optimal parameters - Beta: {beta_opt}, Gamma: {gamma_opt}")

# Run simulation with optimal parameters
solution = odeint(sir_model, (S0, I0, R0), t, args=(N, beta_opt, gamma_opt))
S, I, R = solution.T

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(us_total_cases_wave2.index, I, 'r', label='Infected (model)')
plt.plot(us_total_cases_wave2.index, us_new_cases_wave2, 'b', label='Infected (data)')
plt.xlabel('Date')
plt.ylabel('Number of Infected People')
plt.title('SIR Model vs Actual Data')
plt.legend()
plt.show()
