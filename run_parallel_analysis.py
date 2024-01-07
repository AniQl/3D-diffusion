from src.diffusion_model_fitter import DiffusionModelFitter
import concurrent.futures


def process_file(file, small_delta, Delta):
    fitter = DiffusionModelFitter(file, small_delta, Delta)
    popt_avg = fitter.fit_average_E_model()
    popt_exp = fitter.fit_two_exponential_model()
    csv_filename = f'results/results_{file.split("/")[-1].replace(".txt", ".csv")}'
    fitter.create_csv(popt_avg, popt_exp, csv_filename)
    return f"CSV file created for {file}: {csv_filename}"

data_files = [
    ('data/data_95C.txt', 0.001, 1.3995),
    ('data/data_100C.txt', 0.001, 1.3995),
    # Add more files and parameters as needed
]

# Using ThreadPoolExecutor to parallelize
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_file, file, small_delta, Delta)
               for file, small_delta, Delta in data_files]

    for future in concurrent.futures.as_completed(futures):
        print(future.result())