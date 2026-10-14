# MecaLab

MecaLab is a small mechanical engineering calculation app built with Streamlit. Its main goal is to help users quickly interpret material test data, estimate elastic properties and strength values, and visualize stress-strain behavior from raw CSV files or simple hand calculations.

The project focuses on the mechanical side of the problem rather than the graphical interface itself: it converts experimental data into useful engineering quantities such as:

- stress and strain,
- elastic modulus (Young's modulus),
- yield strength through the 0.2% offset method,
- strength-based summaries for traction and compression,
- approximate properties for unidirectional composites.

## Project goal

The core idea of MecaLab is to make mechanical analysis accessible and fast for common material testing workflows:

1. Input a test result in CSV form.
2. Select the columns corresponding to strain (or displacement) and stress (or force).
3. Convert raw data into stress-strain curves.
4. Estimate elastic parameters and yield point.
5. Compare different curves and export the plots.

This is useful for quick engineering checks, teaching, or preliminary material characterization.

## Mechanical basis

### 1. Stress and strain

For a rectangular specimen, the nominal stress is computed as:

- `σ = F / A0`
- where `A0 = width × thickness`

The strain is computed as:

- `ε = ΔL / L0`

If the input data is already in stress and strain units, no conversion is needed. If the user provides force and displacement, the app converts them before plotting and analysis.

### 2. Young's modulus

The app estimates the elastic modulus from a linear fit of the selected elastic portion of the curve:

- `σ = E × ε + b`

This is done with a least-squares linear regression using `numpy.polyfit(...)` on the chosen strain window. In the code, the relevant section is selected by a user-defined strain range (`strain_min` and `strain_max`). The slope of the fitted line is the Young's modulus, and the intercept is the corresponding offset at zero strain.

### 3. 0.2% offset yield criterion

The function `calculate_offset_yield()` is the most important mechanical routine in this project. It implements the common engineering yield criterion used for metals and similar ductile materials:

- draw a line parallel to the elastic slope,
- offset it by `0.002` strain (0.2%),
- find where this offset line intersects the stress-strain curve.

This is equivalent to the standard `Rp0.2` (proof stress at 0.2% plastic strain) value.

The algorithm works as follows:

- Determine the loading direction based on the sign of the strain range.
- Keep only valid, finite data points that are in the relevant direction of loading.
- Sort data by increasing strain along the loading direction.
- Build the offset line:

  `y_offset = E × (x - direction × offset) + b`

- Compute the difference between the actual stress and the offset-line stress:

  `difference = y - y_offset`

- Look for a sign change between consecutive points. A sign change indicates the curve crosses the offset line.
- Interpolate linearly between the two neighboring points to get a more precise intersection.
- Return the corresponding strain and stress pair.

This gives a single yield point `(ε_yield, σ_yield)` that is used to display the `Rp0,2` marker on the curve.

## Main functions and what they do

### `read_uploaded_csv(uploaded_file, delimiter, encoding, decimal_separator)`

This helper reads a CSV file uploaded by the user. It supports:

- automatic delimiter detection,
- semicolon or comma separated files,
- tab-delimited files,
- common text encodings such as UTF-8 and Latin-1.

It converts the raw file into a pandas DataFrame so the rest of the app can work with it numerically.

### `numeric_column(data, column)`

This small helper converts one selected column to numeric values and replaces invalid entries with `NaN` via `pd.to_numeric(..., errors="coerce")`.

This is critical because mechanical files often contain text notes or missing values, and the app should ignore them rather than crash.

### `calculate_offset_yield(strain, stress, young_modulus, intercept, offset=0.002)`

This is the mechanical heart of the project.

It evaluates the intersection between the measured stress-strain data and the offset line parallel to the elastic region, returning the first valid intersection as a tuple:

- `(intersection_strain, intersection_stress)`

If there is no valid crossing, it returns `None`.

This is how the app identifies the yield stress for the `Rp0.2` criterion.

### `show_plot_downloads(figure, filename, key)`

This function creates the interactive Plotly chart and adds export controls:

- interactive graph display in Streamlit,
- PNG image export,
- HTML export of the full Plotly figure.

It is useful for sharing or saving the mechanical curve results.

### `show_quick_calculator()`

A compact calculator for basic mechanics.

It computes:

- cross-sectional area,
- nominal stress,
- strain,
- secant modulus for the entered point.

This is a fast way to verify formulas without loading a CSV.

### `show_tensile_analysis()`

This is the central analysis page of the app.

It allows the user to:

- load one or more CSV files,
- choose which columns are strain/displacement and stress/force,
- convert displacement to strain using `ΔL / L0`,
- convert force to stress using `F / A0`,
- select the elastic range used for fitting,
- calculate the Young's modulus,
- detect the 0.2% offset yield point,
- plot all curves together,
- summarize maximum tensile/compressive strengths and modulus values.

This is the main material testing workflow in the app.

### `show_composite_analysis()`

This section estimates approximate properties of a unidirectional composite from constituent properties.

It uses the following concepts:

- fiber volume fraction `Vf`,
- longitudinal modulus according to a rule-of-mixtures estimate,
- transverse modulus from inverse mixture relations,
- average Poisson ratio,
- shear modulus from inverse mixture relations,
- directional modulus variation as a function of fiber angle.

The app then plots the evolution of elastic properties with angle `θ`, giving a quick understanding of orthotropic material behavior.

### `show_material_strength()`

This section is intended for structural strength calculations such as bending and buckling. At the moment it is still a placeholder, but it is designed to host future calculations for:

- rectangular sections,
- circular sections,
- I-sections,
- different materials and safety conditions.

### `show_help()`

This displays usage notes explaining the interpretation of the calculations, units, and the meaning of the key outputs such as `Young's modulus`, `Rm`, `Rc`, and `Rp0.2`.

## Typical workflow in the app

1. Load one or more test files.
2. Select the correct columns for strain/displacement and stress/force.
3. Choose units (MPa, kPa, Pa, GPa, etc.).
4. Select an elastic window for the linear regression.
5. Enable the offset analysis to compute `Rp0.2`.
6. Inspect the curve and the resulting metrics table.
7. Export the plot if needed.

## Mechanical interpretation of the results

The most important outputs are:

- `Young's modulus`: stiffness in the elastic region,
- `Rm`: maximum tensile stress reached,
- `Rc`: magnitude of maximum compressive stress,
- `Rp0.2`: engineering yield stress by the 0.2% offset method.

These values are used to compare materials and assess whether a specimen behaves as expected under load.

## Summary

MecaLab is a focused mechanical analysis tool for rapid calculation and visualisation of tensile/compression behavior, elastic parameters, and yield points. The mechanics are intentionally simple and engineering-oriented: it emphasizes practical interpretation over high-end finite-element modeling.

The key mathematical engine is the elastic fit and the 0.2% offset yield detection (`calculate_offset_yield()`), which gives a clear and standard estimate of material yield stress from experimental curves.
