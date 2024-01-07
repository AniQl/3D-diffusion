import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import dblquad
import pandas as pd
import time


class DiffusionModelFitter:
    def __init__(self, filename, small_delta, Delta):
        self.gamma = 103.976E6  # for lithium in Hz/T
        self.small_delta = small_delta  # in seconds
        self.Delta = Delta  # in seconds
        self.data = np.loadtxt(filename, delimiter='\t')
        self.g_values, self.E_values = self._process_data()

    def _process_data(self):
        g_values = self.data[:, 0] / 100  # Convert from Gauss/cm to Tesla/m
        E_values = self.data[:, 1] / np.max(self.data[:, 1])  # Normalize E values
        return g_values, E_values

    def average_E_model(self, g, D1, D2, D3):
        def integrand(alpha, beta, g_val):
            term1 = 2 * D1 * self.Delta * (np.sin(beta) * np.sin(alpha)) ** 2
            term2 = 2 * D2 * self.Delta * (-np.cos(alpha) * np.sin(beta)) ** 2
            term3 = 2 * D3 * self.Delta * (np.cos(beta)) ** 2
            return np.exp(-(self.gamma * self.small_delta * g_val) ** 2 * (term1 + term2 + term3)) * np.sin(beta)

        results = np.empty_like(g)
        for i, g_val in enumerate(g):
            integral, _ = dblquad(integrand, 0, np.pi, lambda beta: 0, lambda beta: 2 * np.pi, args=(g_val,))
            results[i] = integral / (4 * np.pi)
        return results

    def fit_average_E_model(self, initial_guess=[1E-11, 1E-13, 1E-14]):
        start_time = time.time()
        popt_avg, pcov_avg = curve_fit(self.average_E_model, self.g_values, self.E_values, p0=initial_guess, maxfev=2000)
        end_time = time.time()
        print(f"Time taken for average E model fitting: {end_time - start_time} seconds")
        return popt_avg

    def two_exponential_model(self, g, D1exp, D2exp, A1, A2):
        term1 = A1 * np.exp(-(self.gamma * self.small_delta * g) ** 2 * self.Delta * D1exp * (self.Delta - self.small_delta / 3))
        term2 = A2 * np.exp(-(self.gamma * self.small_delta * g) ** 2 * self.Delta * D2exp * (self.Delta - self.small_delta / 3))
        return term1 + term2

    def fit_two_exponential_model(self, initial_guess_exp=[1e-13, 1e-14, 0.5, 0.5]):
        start_time = time.time()
        popt_exp, pcov_exp = curve_fit(self.two_exponential_model, self.g_values, self.E_values, p0=initial_guess_exp, maxfev=2000)
        end_time = time.time()
        print(f"Time taken for two-exponential model fitting: {end_time - start_time} seconds")
        return popt_exp

    def calculate_fit_results(self, popt_avg, popt_exp):
        g_fit = np.linspace(min(self.g_values), max(self.g_values), 500)
        E_fit_avg = self.average_E_model(g_fit, *popt_avg)
        E_fit_exp = self.two_exponential_model(g_fit, *popt_exp)
        return g_fit, E_fit_avg, E_fit_exp

    def plot_results(self, popt_avg, popt_exp, show_plot=False):
        g_fit, E_fit_avg, E_fit_exp = self.calculate_fit_results(popt_avg, popt_exp)

        if show_plot:
            plt.figure(figsize=(10, 6))
            plt.scatter(self.g_values, self.E_values, label='Original Data')
            plt.plot(g_fit, E_fit_avg, color='green', label='Average Model Fit')
            plt.plot(g_fit, E_fit_exp, color='blue', label='Two-Exponential Fit')
            plt.xlabel('g (T/m)')
            plt.ylabel('Normalized E')
            plt.title('Comparison of Fitting Models')
            plt.legend()
            plt.grid(True)
            plt.show(block=False)

    def create_csv(self, popt_avg, popt_exp, filename):
        g_fit, E_fit_avg, E_fit_exp = self.calculate_fit_results(popt_avg, popt_exp)

        # Data for g_fit, E_fit_exp, E_fit_avg
        df_fit = pd.DataFrame({
            'g_fit': g_fit,
            'E_fit_exp': E_fit_exp,
            'E_fit_avg': E_fit_avg
        })

        # Original data
        grad_Gs_cm = self.data[:, 0]
        intensity = self.data[:, 1]
        grad_T_m = grad_Gs_cm / 100
        normalized_intensity = intensity / np.max(intensity)

        df_original = pd.DataFrame({
            'Grad Gs/cm': grad_Gs_cm,
            'Intensity a.u.': intensity,
            'Grad T/m': grad_T_m,
            'Normalized Intensity': normalized_intensity
        })

        # Combine the two dataframes
        max_length = max(len(df_fit), len(df_original))
        df_fit = df_fit.reindex(range(max_length))
        df_original = df_original.reindex(range(max_length))

        # Concatenate DataFrames side by side
        df_combined = pd.concat([df_fit, df_original], axis=1)

        # Save to CSV
        df_combined.to_csv(filename, index=False)

        D1exp_fit, D2exp_fit, A1_fit, A2_fit = popt_exp
        D1_fit_avg, D2_fit_avg, D3_fit_avg = popt_avg
        # Create a DataFrame for model parameters and save to a separate CSV
        df_params = pd.DataFrame({
            'Parameter': ['D1', 'D2', 'D3', 'D1exp', 'D2exp', 'A1', 'A2'],
            'Value': [D1_fit_avg, D2_fit_avg, D3_fit_avg, D1exp_fit, D2exp_fit, A1_fit, A2_fit]
        })

        params_filename = filename.replace('.csv', '_D.csv')
        df_params.to_csv(params_filename, index=False)
        print(f"Model parameters saved to {params_filename}")
