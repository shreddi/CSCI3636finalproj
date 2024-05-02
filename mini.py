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
