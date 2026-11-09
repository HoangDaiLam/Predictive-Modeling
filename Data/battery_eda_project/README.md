# Synthetic Battery EDA Starter Project

This project creates made-up EV trip-segment data and plots exploratory data analysis (EDA) graphs.

**Important:** the data are simulated for coding and modelling practice. They are not measured Polestar 4 data, and the illustrative formula is not a validated vehicle-energy model. Factor choices are based on the project factor list supplied by the user; the synthetic relationships and numeric coefficients are assumptions for demonstration.

## 1. Install dependencies

Python 3.10+ is recommended.

```bash
python -m pip install -r requirements.txt
```

## 2. Generate data

Default: 10,000 rows.

```bash
python generate_battery_data.py
```

Choose the number of observations, output file, and random seed:

```bash
python generate_battery_data.py --rows 50000 --output data/battery_data.csv --seed 7
```

Same seed + same options gives repeatable data.

## 3. Create EDA graphs

```bash
python eda_battery_data.py --input data/battery_data.csv --output-dir reports/eda
```

The script saves PNG graphs:
1. Consumption distribution
2. Speed vs consumption
3. Outside temperature vs consumption
4. Consumption by traffic situation
5. Numeric correlation matrix
6. Consumption by road condition

It also prints the Polars schema, summary statistics, and basic null/negative-energy checks.

## 4. Main target

`consumption_kwh_per_100km` is the main example target column. `energy_used_kwh` is the estimated energy used for the simulated segment.

## 5. Row meaning and limitations

One row represents one synthetic driving segment, not one second of telemetry and not necessarily a complete trip. Because the generator uses a made-up rule, EDA will reveal patterns deliberately introduced by that rule. It cannot establish how the real vehicle behaves.

## Files

- `generate_battery_data.py` — creates the synthetic CSV.
- `eda_battery_data.py` — reads the CSV and creates graphs.
- `requirements.txt` — Python dependencies.
