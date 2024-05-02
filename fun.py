import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def load_data():
    total_cases = pd.read_csv('total_cases.csv', parse_dates=['date'])
    new_cases = pd.read_csv('biweekly_cases.csv', parse_dates=['date'])
    vaccinations = pd.read_csv('USvaccinations.csv', parse_dates=['date'])
    return total_cases.set_index('date'), new_cases.set_index('date'), vaccinations.set_index('date')

def fit_sir_model(total_cases):
    # Assuming the US Population
    N = 331000000  
    # Check data for any missing or infinite values and fill or clean them
    total_cases['United States'] = total_cases['United States'].fillna(method='ffill').replace([np.inf, -np.inf], np.nan).dropna()

    # Initial number of infected and recovered (arbitrary small number of recoveries initially)
    I0 = total_cases['United States'].iloc[0]
    R0 = 0  
    S0 = N - I0 - R0  

    # Make sure initial conditions are finite
    if not np.isfinite(S0) or not np.isfinite(I0) or not np.isfinite(R0):
        raise ValueError("Initial conditions must be finite numbers.")

    print(f"Initial conditions: S0={S0}, I0={I0}, R0={R0}")

    t = np.arange(len(total_cases))
    
    def sir_model(t, y, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]

    def objective(params):
        beta, gamma = params
        sol = solve_ivp(sir_model, [t.min(), t.max()], [S0, I0, R0], args=(beta, gamma), t_eval=t)
        # Root mean square error with respect to recovered
        return np.sqrt(np.mean((sol.y[2] - total_cases['United States'])**2))

    # Optimize parameters
    params_guess = [0.4, 0.1]
    bounds = [(0.0001, 1), (0.0001, 0.2)]
    result = minimize(objective, params_guess, bounds=bounds, method='L-BFGS-B')
    beta_opt, gamma_opt = result.x
    print(f"Optimized parameters: beta={beta_opt}, gamma={gamma_opt}")

    # Plot results for visualization
    optimal_sol = solve_ivp(sir_model, [t.min(), t.max()], [S0, I0, R0], args=(beta_opt, gamma_opt), t_eval=t)
    plt.figure(figsize=(10, 6))
    plt.plot(total_cases.index, total_cases['United States'], 'r', label='Total Cases')
    plt.plot(total_cases.index, optimal_sol.y[2], 'b', label='Fitted Recovered')
    plt.xlabel('Date')
    plt.ylabel('Number of Cases')
    plt.title('Fit of SIR Model to COVID-19 Data')
    plt.legend()
    plt.show()

total_cases, new_cases, vaccinations = load_data()
fit_sir_model(total_cases)
