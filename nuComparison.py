import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import minimize

"""
Load Data
"""
#Total cumulative cases
total_cases_data = pd.read_csv('total_cases.csv')
total_cases_data['date'] = pd.to_datetime(total_cases_data['date'])
total_cases_data.set_index('date', inplace=True)
us_total_cases = total_cases_data['United States']

#Estimated recoveries, by offsetting total cases by 14 days
recovery_delay = pd.DateOffset(days=14)
us_recovered = us_total_cases.shift(-recovery_delay.days)

#New cases, counted biweekly
new_cases_data = pd.read_csv('biweekly_cases.csv')
new_cases_data['date'] = pd.to_datetime(new_cases_data['date'])
new_cases_data.set_index('date', inplace=True)
us_new_cases = new_cases_data['United States']

#Vaccines administered
vax_data = pd.read_csv('vaccinations.csv')
vax_data['date'] = pd.to_datetime(vax_data['date'])
vax_data.set_index('date', inplace=True)
us_vax_data = vax_data[(vax_data['location'] == 'United States')]

#Population of countries in the world
population_data = pd.read_csv('population_data.csv')
population_2020 = population_data[(population_data['Time'] == 2020) & (population_data['Variant'] == 'Medium')]





"""
Define Wave, and filter data for said wave
"""
wave1 = True
# wave1 = False

if(wave1==True):
    wave_start = '2020-03-01'
    wave_end = '2021-07-01'
else:
    wave_start = '2021-07-01'
    wave_end = '2023-02-01'

#Filter data for the first wave period
us_total_cases_wave = us_total_cases[wave_start:wave_end]
us_new_cases_wave = us_new_cases[wave_start:wave_end]
us_recovered_wave = us_recovered[wave_start:wave_end]
us_vax_wave = us_vax_data.loc[wave_start:wave_end, 'people_fully_vaccinated']





"""
Find Nu values for several different countries
"""
countries = ['Germany', 'India', 'Argentina', 'Japan', 'Canada', 'United Kingdom']
nu_values = {}

#Print normalized nu values
for country, nu in nu_values.items():
    print(f"Normalized average nu for {country}: {nu:.6f}")

#Print the nu values for each country
for country, nu in nu_values.items():
    print(f"Average nu for {country}: {nu:.6f}")
    
for country in countries:
    country_population = population_2020[population_2020['Location'] == country]['PopTotal'].iloc[0] * 1000  #Population in thousands
    country_vax_data = vax_data[(vax_data['location'] == country)].fillna(0)
    country_vax_wave = country_vax_data.loc[wave_start:wave_end, 'people_fully_vaccinated'].fillna(0)
    V0 = country_vax_wave.iloc[0] if not country_vax_wave.empty else 0
    vax_diff = country_vax_wave.diff().fillna(0)
    
    #Normalizing Nu by the initial susceptible population
    total_susceptible_start = country_population - V0
    nu_avg_normalized = (vax_diff / total_susceptible_start).clip(lower=0).mean()  #Normalized average nu
    nu_values[country] = nu_avg_normalized

nu_values['No Vaccinations'] = 0



"""
Define parameters for SIR and SVIR Model using data for the United States
"""
N = population_2020[population_2020['Location'] == 'United States']['PopTotal'].iloc[0] * 1000  #Approximate US population in 2020
I0 = us_new_cases_wave.iloc[0]  #Initial number of infected individuals
V0 = us_vax_wave.iloc[0]
R0 = 0  #Initial number of recovered individuals
S0 = N - I0 - R0  #Initial susceptible population

#Calculate daily changes in vaccination (approximate daily vaccination rate)
vax_diff = us_vax_wave.diff().fillna(0)
nu_avg = (vax_diff / S0).clip(lower=0).mean()  #Average nu over the period for simplicity





"""
Define SIR and SVIR Model
"""
#SIR Model
def sir_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

#SVIR Model
def svir_model(y, t, N, beta, gamma, nu):
    S, V, I, R = y
    dSdt = -beta * S * I / N - nu * S
    dVdt = nu * S
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dVdt, dIdt, dRdt





"""
Fit models to real-world data
"""
#Time vector for simulation
t = np.arange(len(us_new_cases_wave))

#Fit the SIR model
result_sir = minimize(lambda x: np.mean((odeint(sir_model, (S0, I0, R0), t, args=(N, *x))[:, 1] - us_new_cases_wave.values)**2),
                      [0.1, 0.1], method='L-BFGS-B', bounds=[(0.0001, 1), (0.0001, 1)])
beta_opt_sir, gamma_opt_sir = result_sir.x

#Fit the SVIR model
result_svir = minimize(lambda x: np.mean((odeint(svir_model, (S0, 0, I0, R0), t, args=(N, *x, nu_avg))[:, 2] - us_new_cases_wave.values)**2),
                       [0.1, 0.1], method='L-BFGS-B', bounds=[(0.0001, 1), (0.0001, 1)])
beta_opt_svir, gamma_opt_svir = result_svir.x

#Function to calculate sum of squared errors
def calculate_mse(observed, predicted):
    """Calculate the mean squared error."""
    return np.mean((observed - predicted) ** 2)

#Run simulation with optimal parameters and calculate SSE
solution_sir = odeint(sir_model, (S0, I0, R0), t, args=(N, beta_opt_sir, gamma_opt_sir))
I_sir = solution_sir[:, 1]
sse_sir = calculate_mse(us_new_cases_wave.values, I_sir)

solution_svir = odeint(svir_model, (S0, V0, I0, R0), t, args=(N, beta_opt_svir, gamma_opt_svir, nu_avg))
I_svir = solution_svir[:, 2]
sse_svir = calculate_mse(us_new_cases_wave.values, I_svir)





"""
Plot data
"""
#Nu variations for comparison
colors = ['red', 'blue', 'green', 'purple', 'orange', 'pink', 'brown']  #Assign a color for each country for plotting
color_iter = iter(colors)  #Create an iterator over the colors list

plt.figure(figsize=(10, 6))

for country, nu in nu_values.items():
    color = next(color_iter)  #Get the next color from the iterator

    #Assuming beta_opt_svir and gamma_opt_svir are defined from your model fitting
    #Run simulation with optimal parameters for the current country's nu
    solution = odeint(svir_model, (S0, V0, I0, R0), t, args=(N, beta_opt_svir, gamma_opt_svir, nu))
    S, V, I, R = solution.T

    #Plot the results
    plt.plot(us_new_cases_wave.index, I, label=f'{country} Nu={nu}', color=color)

plt.plot(us_new_cases_wave.index, us_new_cases_wave, 'y', label='Infected (Actual)')
plt.plot(us_new_cases_wave.index, I_svir, 'k', label=f'United States Nu = {nu_avg}')
plt.xlabel('Date')
plt.ylabel('Number of Infected People')
plt.title('SVIR Model Predictions with Different Nu Values')
plt.legend()
plt.show()
