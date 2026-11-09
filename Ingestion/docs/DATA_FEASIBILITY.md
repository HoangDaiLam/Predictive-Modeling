# Data and implementation feasibility assessment

## Dataset scale
The generated release contains 10,000 trip-level observations and 45 columns. The CSV is approximately a few megabytes, so it is straightforward to store in Git (subject to the team's repository rules), share through cloud storage, and load into Pandas/scikit-learn on an ordinary laptop. A database is not justified by this scale alone.

## Observed release statistics (seed 42, v1)
The current generated release should be regenerated after code changes; the committed quality report and CSV are the authoritative outputs for that release.

- Trip distance is bounded between 2 and 240 km.
- Average speed is bounded between 8 and 120 km/h by construction.
- Duration is derived as distance / average speed, not generated independently.
- The data-quality gate checks the speed identity and the normalized energy identity for every row.
- Energy consumption is generated from wheel road-load work, drivetrain conversion, representative stop-start cycles, regenerative recovery, HVAC and auxiliary loads.

## Runtime feasibility
- Dataset generation: vectorized NumPy/Pandas; 10,000 rows runs quickly on a standard laptop.
- API: read-only FastAPI service; paginated responses prevent unnecessarily large JSON payloads.
- ML: 10,000 rows is a small tabular dataset for common regression models. Use cross-validation and a final untouched test set.
- Sharing: commit the canonical CSV if acceptable; otherwise distribute a versioned release file via the team's shared drive and keep the version/seed in Git.

## Scientific feasibility and limitations
This is synthetic data generated from a simplified model. It is appropriate for developing the data pipeline, evaluator dashboard, model comparison, and feature engineering workflow. It cannot establish real-world predictive accuracy without measured EV trip data. Average-speed aerodynamic work is an approximation: true aerodynamic energy depends on the time integral of relative air speed cubed, which cannot be reconstructed from one trip-average speed. Battery capacity and health are contextual columns, not causal drivers of this model's energy-demand label. Do not claim that the generated data is a validated vehicle digital twin.


## Dashboard feasibility
Streamlit is suitable for this project because the dashboard is an internal team tool and the dataset is small. Generation is vectorized with NumPy/Pandas and should remain responsive at 10,000 rows. FastAPI and Streamlit run as separate processes: the dashboard can generate the local canonical CSV even when the API is offline, while API inspection requires FastAPI to be running. For much larger data or concurrent multi-user writes, move generation to a background job and introduce stronger dataset-release coordination; that is not necessary for the current scale.
