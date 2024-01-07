from src.diffusion_model_fitter import DiffusionModelFitter

data_files = [
    ('data/data_95C.txt', 0.001, 1.3995),
    ('data/data_100C.txt', 0.001, 1.3995),
    # Add more files and parameters as needed
]

for file, small_delta, Delta in data_files:
    # Initialize the fitter with the current file and parameters
    fitter = DiffusionModelFitter(file, small_delta, Delta)

    # Fit the models
    popt_avg = fitter.fit_average_E_model()
    popt_exp = fitter.fit_two_exponential_model()

    # Optionally, you can plot the results (set show_plot=True if needed)
    # fitter.plot_results(popt_avg, popt_exp, show_plot=False)

    # Create a CSV file for this run
    csv_filename = f'results_{file.replace(".txt", "")}.csv'
    fitter.create_csv(popt_avg, popt_exp, csv_filename)

    print(f"CSV file created for {file}: {csv_filename}")