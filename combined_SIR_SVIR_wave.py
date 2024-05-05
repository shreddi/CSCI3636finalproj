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
us_recovered = us_total_cases.shift(-recovery_delay.days)

new_cases_data = pd.read_csv('biweekly_cases.csv')
new_cases_data['date'] = pd.to_datetime(new_cases_data['date'])
new_cases_data.set_index('date', inplace=True)
us_new_cases = new_cases_data['United States']

vax_data = pd.read_csv('vaccinations.csv')
vax_data['date'] = pd.to_datetime(vax_data['date'])
vax_data.set_index('date', inplace=True)
us_vax_data = vax_data[(vax_data['location'] == 'United States')]

wave1 = True
# wave1 = False

if(wave1==True):
    wave_start = '2020-03-01'
    wave_end = '2021-07-01'
else:
    wave_start = '2021-07-01'
    wave_end = '2023-02-01'

# Filter data for the first wave period
us_total_cases_wave = us_total_cases[wave_start:wave_end]
us_new_cases_wave = us_new_cases[wave_start:wave_end]
us_recovered_wave = us_recovered[wave_start:wave_end]
us_vax_wave = us_vax_data.loc[wave_start:wave_end, 'people_fully_vaccinated']


# Define parameters for SIR and SVIR Model
N = 331000000  # Approximate US population in 2020
I0 = us_new_cases_wave.iloc[0]  # Initial number of infected individuals
V0 = us_vax_wave.iloc[0]
R0 = 0  # Initial number of recovered individuals
S0 = N - I0 - R0  # Initial susceptible population

# Calculate daily changes in vaccination (approximate daily vaccination rate)
vax_diff = us_vax_wave.diff().fillna(0)
nu_avg = (vax_diff / S0).clip(lower=0).mean()  # Average nu over the period for simplicity

# List of countries of interest
countries = ['Germany', 'India', 'Brazil']

# Dictionary to hold nu values for each country
nu_values = {}

# SIR Model
def sir_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# SVIR Model
def svir_model(y, t, N, beta, gamma, nu):
    S, V, I, R = y
    dSdt = -beta * S * I / N - nu * S
    dVdt = nu * S
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dVdt, dIdt, dRdt

# Time vector for simulation
t = np.arange(len(us_new_cases_wave))

# Fit the SIR model
result_sir = minimize(lambda x: np.mean((odeint(sir_model, (S0, I0, R0), t, args=(N, *x))[:, 1] - us_new_cases_wave.values)**2),
                      [0.1, 0.1], method='L-BFGS-B', bounds=[(0.0001, 1), (0.0001, 1)])
beta_opt_sir, gamma_opt_sir = result_sir.x

# Fit the SVIR model
result_svir = minimize(lambda x: np.mean((odeint(svir_model, (S0, 0, I0, R0), t, args=(N, *x, nu_avg))[:, 2] - us_new_cases_wave.values)**2),
                       [0.1, 0.1], method='L-BFGS-B', bounds=[(0.0001, 1), (0.0001, 1)])
beta_opt_svir, gamma_opt_svir = result_svir.x

# Function to calculate sum of squared errors
def calculate_mse(observed, predicted):
    """Calculate the mean squared error."""
    return np.mean((observed - predicted) ** 2)

# Run simulation with optimal parameters and calculate SSE
solution_sir = odeint(sir_model, (S0, I0, R0), t, args=(N, beta_opt_sir, gamma_opt_sir))
I_sir = solution_sir[:, 1]
sse_sir = calculate_mse(us_new_cases_wave.values, I_sir)

solution_svir = odeint(svir_model, (S0, V0, I0, R0), t, args=(N, beta_opt_svir, gamma_opt_svir, nu_avg))
I_svir = solution_svir[:, 2]
sse_svir = calculate_mse(us_new_cases_wave.values, I_svir)

def plot():
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    # SIR Plot
    axs[0].plot(us_new_cases_wave.index, I_sir, 'r', label='Infected (SIR model)')
    axs[0].plot(us_new_cases_wave.index, us_new_cases_wave, 'b', label='Infected (data)')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Number of Infected People')
    axs[0].set_title('SIR Model vs Actual Data')
    axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    axs[0].text(0.1, 0.95, f'Beta: {beta_opt_sir:.4f}\nGamma: {gamma_opt_sir:.4f}\nMSE: {sse_sir:.2e}', 
                transform=axs[0].transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left')
    axs[0].tick_params(axis='x', labelrotation=45)  # Rotate x-axis labels

    # SVIR Plot
    axs[1].plot(us_new_cases_wave.index, I_svir, 'r', label='Infected (SVIR model)')
    axs[1].plot(us_new_cases_wave.index, us_new_cases_wave, 'b', label='Infected (data)')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Number of Infected People')
    axs[1].set_title('SVIR Model vs Actual Data')
    axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
    axs[1].text(0.1, 0.95, f'Beta: {beta_opt_svir:.4f}\nGamma: {gamma_opt_svir:.4f}\nNu: {nu_avg:.4e}\nMSE: {sse_svir:.2e}', 
                transform=axs[1].transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left')
    axs[1].tick_params(axis='x', labelrotation=45)  # Rotate x-axis labels

    plt.show()

plot()



