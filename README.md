# Diffusion Model Fitter

## Overview
`DiffusionModelFitter` is a Python class designed for fitting diffusion models to data. This class is particularly tailored for analyzing diffusion in lithium using two different models: an average E model and a two-exponential model.

## Requirements
- Python 3
- NumPy
- Matplotlib
- SciPy
- Pandas

## Features
- Loading and normalization of experimental data.
- Fitting using the average E model and the two-exponential model.
- Visualization of the original data and fitted curves.
- Exporting fitted data to a CSV file.

## Usage

### Initialization
First, create an instance of the `DiffusionModelFitter` by passing the filename of your data, along with `small_delta` and `Delta` values:
```python
fitter = DiffusionModelFitter('data_100C.txt', 0.001, 1.3995)
```
### Fitting Models
To fit the average E model and the two-exponential model, use:
```python
popt_avg = fitter.fit_average_E_model()
popt_exp = fitter.fit_two_exponential_model()
```

### CSV File Format
The CSV file will contain the following columns:

- `g_fit`: The g-values used for fitting.
- `E_fit_exp`: The E values from the two-exponential model.
- `E_fit_avg`: The E values from the average E model.
- `Grad Gs/cm`: The gradient in Gauss/cm, as provided in the original data.
- `Intensity a.u.`: The intensity values, as provided in the original data.
- `Grad T/m`: The gradient converted to Tesla/m.
- `Normalized Intensity`: The normalized intensity values.