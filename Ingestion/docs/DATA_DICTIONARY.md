# Data dictionary

Each row is one complete synthetic trip. Units appear in column names.

## Identifiers and vehicle
- `sample_id`: unique trip/sample ID; exclude from ML features.
- `dataset_version`: release label; exclude from ML features.
- `random_seed`: seed used to generate this release; exclude from ML features.
- `vehicle_mass_kg`: representative vehicle plus payload mass.
- `battery_capacity_kwh`: nominal battery capacity.
- `battery_health_fraction`: illustrative state-of-health fraction.
- `drag_coefficient`, `frontal_area_m2`, `rolling_resistance_coefficient`: vehicle and tyre parameters.
- `motor_efficiency`, `inverter_efficiency`, `mechanical_efficiency`: component efficiencies.
- `regenerative_braking_efficiency`: fraction of braking kinetic energy recovered.
- `drivetrain_mode`: eco, normal or sport.

## Route and driving
- `road_type`, `road_surface`: categorical road descriptors.
- `distance_km`: trip distance.
- `duration_minutes`: elapsed trip duration, calculated from distance and average speed.
- `average_speed_kmh`: distance divided by elapsed trip time.
- `speed_limit_kmh`: posted/synthetic speed limit.
- `traffic_index`: 0–1 congestion indicator.
- `net_grade_percent`: signed net elevation change divided by distance × 100.
- `elevation_change_m`: signed destination-minus-origin elevation.
- `curvature_index`: illustrative 0–1 route curvature descriptor.
- `stop_count`: estimated stop-start events.
- `mean_abs_acceleration_mps2`: illustrative mean absolute acceleration descriptor.
- `acceleration_event_speed_mps`: representative speed used for stop-start kinetic energy.

## Weather and auxiliary loads
`ambient_temperature_c`, `snow_depth_cm`, `precipitation_mm_h`, `wind_speed_mps`, `wind_direction_factor`, `hvac_mode`, `hvac_power_kw`, `auxiliary_power_kw`.

## Energy-model audit columns
- `rolling_energy_wheel_kwh`, `aerodynamic_energy_wheel_kwh`, `grade_energy_wheel_kwh`: wheel-side work components.
- `transient_energy_battery_kwh`: net battery energy for representative acceleration/braking events.
- `hvac_energy_kwh`, `auxiliary_energy_kwh`: time-dependent battery loads.

## Targets
- `energy_consumption_kwh`: total trip battery energy.
- `consumption_kwh_per_100km`: normalized trip energy target.

For supervised learning, exclude identifiers, dataset version, target columns, and all audit/derived energy-component columns from the feature matrix. Otherwise the model can leak the target through variables calculated directly from it.
