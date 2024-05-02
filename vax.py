import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load the datasets
total_cases_data = pd.read_csv('total_cases.csv')
total_cases_data['date'] = pd.to_datetime(total_cases_data['date'])
total_cases_data.set_index('date', inplace=True)

new_cases_data = pd.read_csv('biweekly_cases.csv')
new_cases_data['date'] = pd.to_datetime(new_cases_data['date'])
new_cases_data.set_index('date', inplace=True)

vax_data = pd.read_csv('USvaccinations.csv')
vax_data['date'] = pd.to_datetime(vax_data['date'])
vax_data.set_index('date', inplace=True)

# Assume the recovered cases by shifting the total cases by a typical recovery period
recovery_delay = pd.DateOffset(days=14)
total_cases_data['Recovered'] = total_cases_data['United States'].shift(-recovery_delay.days)

# Define the SIR model
def SIR(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Initial conditions
N = 331000000  # Total population estimate
I0 = new_cases_data['United States'].iloc[0]
R0 = total_cases_data['United States'].dropna().iloc[0]
S0 = N - I0 - R0
initial_conditions = [S0/N, I0/N, R0/N]

# Time points
t = np.linspace(0, len(new_cases_data) - 1, len(new_cases_data))

# Solve the SIR model with optimized parameters
def simulate_SIR(beta, gamma):
    return odeint(SIR, initial_conditions, t, args=(beta, gamma))

# Objective function for optimization
def objective(params):
    beta, gamma = params
    sim = simulate_SIR(beta, gamma)
    sim_R = sim[:, 2] * N  # Recovered
    real_R = total_cases_data['United States'].dropna()
    return np.mean((sim_R[:len(real_R)] - real_R)**2)

# Optimization
params_guess = [0.4, 0.1]
bounds = [(0.0001, 1), (0.0001, 0.2)]
result = minimize(objective, params_guess, bounds=bounds, method='L-BFGS-B')
beta_opt, gamma_opt = result.x

# Get the optimized simulation results
sim_results = simulate_SIR(beta_opt, gamma_opt)
sim_S, sim_I, sim_R = sim_results.T * N  # Scale back up by the population

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(new_cases_data.index, new_cases_data['United States'], label='New Cases')
plt.plot(total_cases_data.index, total_cases_data['United States'], label='Total Cases')
plt.plot(total_cases_data.index, total_cases_data['Recovered'], label='Estimated Recovered', linestyle='--')
plt.plot(total_cases_data.index[:len(sim_I)], sim_I, label='Fitted Infected', color='blue', linestyle=':')
plt.plot(total_cases_data.index[:len(sim_R)], sim_R, label='Fitted Recovered', color='green', linestyle=':')
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('COVID-19 Data with SIR Model Fit')
plt.legend()
plt.show()
