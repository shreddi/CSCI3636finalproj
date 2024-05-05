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

wave1_start = '2020-03-01'
wave1_end = '2021-07-01'

# Filter data for the first wave period
us_total_cases_wave1 = us_total_cases[wave1_start:wave1_end]
us_new_cases_wave1 = us_new_cases[wave1_start:wave1_end]
us_recovered_wave1 = us_recovered[wave1_start:wave1_end]
us_vax_wave1 = vax_data.loc[wave1_start:wave1_end, 'people_fully_vaccinated']

# Calculate daily changes in vaccination (approximate daily vaccination rate)
vax_diff = us_vax_wave1.diff().fillna(0)

N = 331000000  # Approximate US population in 2020
I0 = us_new_cases_wave1.iloc[0]  # Initial number of infected individuals
R0 = 0  # Initial number of recovered individuals
V0 = 0  # Initial number of vaccinated individuals, no one is vaccinated
S0 = N - I0 - R0 - V0  # Initial susceptible population
nu_avg = (vax_diff / S0).clip(lower=0).mean()
print(nu_avg)

# Define the SVIR model
def svir_model(y, t, N, beta, gamma, nu):
    S, V, I, R = y
    dSdt = -beta * S * I / N - nu * S
    dVdt = nu * S
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dVdt, dIdt, dRdt

# Function to integrate the model
def fit_svir_odeint(x):
    beta, gamma, nu = x
    return odeint(svir_model, (S0, V0, I0, R0), t, args=(N, beta, gamma, nu))[:, 2]  # Returning only Infected

# Total days for simulation
t = np.arange(len(us_new_cases_wave1))

# Optimization to fit the model to the new case data
result = minimize(lambda x: np.mean((fit_svir_odeint(x) - us_new_cases_wave1.values)**2),
                  [0.1, 0.1, nu_avg], method='L-BFGS-B', bounds=[(0.0001, 1), (0.0001, 1), (nu_avg,nu_avg)]) 
beta_opt, gamma_opt, nu_opt = result.x

print(beta_opt, gamma_opt, nu_opt)

# Run simulation with optimized parameters
solution = odeint(svir_model, (S0, V0, I0, R0), t, args=(N, beta_opt, gamma_opt, nu_avg))
S, V, I, R = solution.T

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(us_new_cases_wave1.index, I, 'r', label='Infected (model)')
plt.plot(us_new_cases_wave1.index, us_new_cases_wave1, 'b', label='Infected (data)')
plt.xlabel('Date')
plt.ylabel('Number of Infected People')
plt.title('SVIR Model vs Actual Data')
plt.legend()
plt.show()