# Predictive Modelling Project — Data Collection and Preparation Summary

## 1. Define the Objective

The main goal of this week's work is to understand:

- What data our project needs
- Where we can get the data
- What type of data we will work with
- How we will collect and process the data
- How we can use the processed data
- How the data will eventually be used for machine learning

For our **Polestar 4 battery-consumption prediction project**, the objective is to predict the vehicle's energy consumption under different driving and environmental conditions.

A possible target variable is:

`Energy Consumption (kWh/km)`

Conceptually:

```
Input conditions
       ↓
Machine Learning Model
       ↓
Predicted Energy Consumption
       ↓
kWh/km
```

---

# 2. What Data Will We Need?

We need to identify variables that may influence energy consumption.

| Category | Factor | What we should collect | Unit / format |
|---|---|---|---|
| **Battery** | Battery level | State of charge (SOC) | % |
| **Driving** | Current driving pattern | Speed, acceleration/braking behaviour | km/h, m/s² |
| **Driving** | Historical driving pattern | Previous driving/consumption behaviour | Time-series |
| **Driving** | Speed | Vehicle speed | km/h |
| **Traffic** | Traffic / city-driving situation | Traffic condition and frequent stopping/starting | Category / measurable values |
| **Temperature** | Outside temperature | Ambient temperature | °C |
| **Battery** | Battery temperature | Traction-battery temperature | °C |
| **Preconditioning** | Vehicle/battery preconditioning | Whether preconditioning is active | On/Off |
| **Climate** | Climate settings | Climate-control usage/settings | On/Off / settings |
| **Tyres** | Tyre condition | Tyre condition | Available measurement/category |
| **Tyres** | Tyre pressure | Tyre pressure | bar |
| **Road** | Road condition | Road surface/condition | Category |
| **Topography** | Slopes/topography | Uphill/downhill/elevation characteristics | % / m |
| **Towing** | Towing | Whether a trailer is being towed | Yes/No |

# 5. What Type of Data Will We Work With?

Our project may contain several different types of data.

## 5.1 Numerical Data

Numerical data contains measurable values.

Examples:

```text
Temperature = -5 °C
Speed = 80 km/h
Wind speed = 15 km/h
Payload = 250 kg
Elevation = 120 m
SOC = 75 %
Tire pressure = 2.8 bar
```

---

## 5.2 Categorical Data

Categorical data represents groups or categories.

Examples:

```text
Weather = Snow
Road = Wet
Driving mode = Standard
Road type = Highway
Traffic = Heavy
```

These categories may need to be converted into numerical representations before machine learning.

---

## 5.3 Boolean Data

Boolean data contains two possible states.

Examples:

```text
AC = ON/OFF
Heating = ON/OFF
Exterior equipment = ON/OFF
Regenerative braking = ON/OFF
```

They can potentially be represented as:

```text
ON  = 1
OFF = 0
```

---

## 5.4 Time-Series Data

Many variables change over time.

Example:

| Time | Speed | SOC | Temperature |
|---|---:|---:|---:|
| 10:00:00 | 50 km/h | 80% | -5°C |
| 10:00:01 | 52 km/h | 80% | -5°C |
| 10:00:02 | 55 km/h | 79% | -5°C |
| 10:00:03 | 58 km/h | 79% | -5°C |

Time-series data is particularly important because vehicle behaviour changes throughout a trip.

---

# 6. Where Can We Get the Data?

For every variable, we need to identify its source.

Potential sources include:

## 6.1 Vehicle Measurements

Potential information:

- Speed
- SOC
- Power consumption
- Battery temperature
- Tire pressure
- HVAC usage
- Regenerative braking

Actual measurements would represent real vehicle behaviour.

---

## 6.2 Weather Data

Potential information:

- Temperature
- Wind speed
- Wind direction
- Precipitation
- Snow
- Humidity
- Weather conditions

---

## 6.3 Geographic and Elevation Data

Potential information:

- GPS coordinates
- Elevation
- Route
- Road gradient

---

## 6.4 Traffic Data

Potential information:

- Traffic density
- Average traffic speed
- Congestion
- Number of stops

---

## 6.5 Vehicle Specifications

Technical documentation can provide information such as:

- Battery capacity
- Vehicle mass
- Motor specifications
- Drivetrain specifications

Vehicle specifications should not be treated as real-world consumption measurements.

---

# 7. Data Source Classification

We should distinguish between different types of data.

## 7.1 Measured Data

Data directly measured from a vehicle, sensor, or experiment.

Example:

```text
Vehicle speed = 72 km/h
Battery SOC = 63%
Power = 24 kW
```

---

## 7.2 External Dataset

Data collected by another organization or system.

Example:

```text
Temperature = -8°C
Wind speed = 14 m/s
```

---

## 7.3 Manufacturer/Technical Data

Information from technical documentation.

Example:

```text
Battery capacity
Vehicle mass
Motor specifications
Vehicle dimensions
```

---

## 7.4 Simulated Data

Data generated using a mathematical or computational model.

Example:

```text
Simulated temperature = -10°C
Simulated payload = 300 kg
Simulated road gradient = 5%
```

Simulated data must be clearly identified as simulated and should not be presented as real vehicle measurements.

---

# 8. How Will We Process the Data?

Raw data cannot immediately be used for machine learning.

The general processing pipeline is:

```text
Raw Data
    ↓
Data Collection
    ↓
Data Integration
    ↓
Data Cleaning
    ↓
Missing-Value Handling
    ↓
Outlier Checking
    ↓
Unit Conversion
    ↓
Categorical Encoding
    ↓
Feature Engineering
    ↓
Exploratory Data Analysis
    ↓
Final Dataset
    ↓
Machine Learning
```

---

# 9. Data Cleaning

We need to check whether the collected data is usable.

## 9.1 Missing Values

Example:

```text
Temperature = -5
Speed = 75
Wind = NULL
SOC = 67
```

We need to determine how missing values should be handled.

Possible approaches include:

- Removing records
- Replacing values
- Estimating values
- Marking values as unavailable

The appropriate method depends on the variable and amount of missing data.

---

## 9.2 Duplicate Data

We need to identify records that have accidentally been collected more than once.

Example:

```text
10:15:23, 75 km/h, -4°C
10:15:23, 75 km/h, -4°C
```

If these are genuine duplicates, they may need to be removed.

---

## 9.3 Incorrect Values

We need to identify unrealistic or technically invalid measurements.

Example:

```text
Temperature = 500°C
Vehicle speed = 900 km/h
Tire pressure = -2 bar
```

Such values must be investigated before being used.

---

# 10. Unit Consistency

Different sources may use different units.

Examples:

```text
Temperature:
°C

Wind:
km/h
m/s

Distance:
km
m

Energy:
Wh
kWh

Power:
W
kW
```

The data needs to be converted into consistent units before datasets are combined.

Examples:

```text
1 kWh = 1000 Wh

1 m/s = 3.6 km/h
```

The project should define the standard unit for every variable.

---

# 11. Data Integration

Our data may come from several sources.

```text
Vehicle Data
      +
Weather Data
      +
GPS/Elevation Data
      +
Traffic Data
      +
Road Data
      ↓
Combined Dataset
```

The important issue is that the datasets must refer to compatible:

- Time
- Location
- Trip
- Vehicle conditions

For example:

```text
Vehicle:
10:30:01 → Speed = 70 km/h

Weather:
10:30:01 → Temperature = -5°C

GPS:
10:30:01 → Elevation = 120 m
```

These observations can potentially be combined because they correspond to the same time.

---

# 12. Feature Engineering

Raw data is not always the most useful form for machine learning.

We may create new variables from the raw data.

## 12.1 Energy Consumption per Kilometre

```text
Energy Consumption =
Energy Used (kWh) / Distance (km)
```

---

## 12.2 Road Gradient

Elevation data can be transformed into road gradient:

```text
Gradient =
Change in Elevation / Horizontal Distance × 100
```

---

## 12.3 Wind Effect

Wind speed alone may not completely describe its effect.

Depending on available data, wind can potentially be represented as:

- Headwind
- Tailwind
- Crosswind

---

## 12.4 Temperature Features

Potential derived variables include:

```text
Temperature
Temperature below 0°C
Temperature below -10°C
```

Feature engineering should be based on the actual available data and modelling objective.

---

# 13. Exploratory Data Analysis

Before machine learning, we need to understand the dataset.

Questions include:

## Does temperature affect consumption?

Compare energy consumption under different temperatures.

```text
Cold temperature
       ↓
Energy consumption
```

---

## Does speed affect consumption?

Compare consumption at different speeds:

```text
30 km/h
50 km/h
80 km/h
100 km/h
120 km/h
```

---

## Does elevation affect consumption?

Compare:

```text
Flat road
Uphill
Downhill
```

---

## Does traffic affect consumption?

Compare:

```text
Low traffic
Medium traffic
High traffic
```

---

## Does HVAC affect consumption?

Compare:

```text
HVAC OFF
HVAC ON
```

The purpose of exploratory analysis is to understand relationships in the data before building the prediction model.

---

# 14. What Can We Do With the Processed Data?

Once the data has been cleaned and prepared, it can be used for several purposes.

## 14.1 Descriptive Analysis

We can calculate and describe:

- Average energy consumption
- Average speed
- Average temperature
- Average power
- Average distance
- Other relevant statistics

---

## 14.2 Relationship Analysis

We can investigate relationships such as:

```text
Temperature → Energy Consumption
Speed → Energy Consumption
Elevation → Energy Consumption
Payload → Energy Consumption
Traffic → Energy Consumption
HVAC → Energy Consumption
```

---

## 14.3 Predictive Modelling

We can train a machine-learning model to estimate:

```text
Energy Consumption = f(Input Variables)
```

---

## 14.4 Range Estimation

Predicted energy consumption can potentially be used to estimate driving range:

```text
Range =
Available Battery Energy /
Predicted Energy Consumption per km
```

The accuracy depends on the quality of the underlying data and assumptions.

---

# 15. Define the Model Inputs and Output

This should be clearly documented.

## Possible Inputs / Features

```text
Speed
Acceleration
Temperature
Wind speed
Wind direction
Elevation
Road gradient
Road quality
Traffic density
Payload
Tire pressure
SOC
Battery temperature
HVAC usage
Regenerative braking
Driving mode
Weather
```

## Possible Output / Target

```text
Energy Consumption (kWh/km)
```

Conceptually:

```text
Temperature ────┐
Speed ──────────┤
Wind ───────────┤
Elevation ──────┤
Traffic ────────┤
Payload ────────┤
HVAC ───────────┤
Tire Pressure ──┤
SOC ────────────┤
                 ↓
       Machine Learning Model
                 ↓
       Energy Consumption
             (kWh/km)
```

---

# 16. Data Quality

Data quality should be treated as a separate part of the project.

We need to check:

## Completeness

Do we have enough observations?

## Accuracy

Are the measurements reliable?

## Consistency

Are units, definitions, and formats consistent?

## Validity

Are values within realistic ranges?

## Timeliness

Does the data correspond to the conditions we want to model?

## Synchronization

Do vehicle, weather, GPS, and traffic data correspond to the same time and location?

---

# 17. Data Limitations

We should document limitations clearly.

Potential limitations include:

```text
Some vehicle variables may not be publicly available.

Some datasets may not be Polestar 4-specific.

Some measurements may need to be simulated.

Different datasets may have different sampling rates.

Some variables may contain missing values.

Some external data may not perfectly represent the actual vehicle conditions.
```

These limitations affect how the final model should be interpreted.

---

# 18. Preparing the Dataset for Machine Learning

After processing, the final dataset should have a structured format.

Example:

| Temperature | Speed | Wind | Elevation | Traffic | Payload | HVAC | Tire Pressure | Consumption |
|---:|---:|---:|---:|---|---:|---|---:|---:|
| -5 | 80 | 15 | 120 | Medium | 200 | On | 2.8 | ... |
| -2 | 60 | 8 | 100 | Low | 150 | Off | 2.8 | ... |
| 5 | 100 | 12 | 80 | High | 250 | Off | 2.7 | ... |

The exact structure will depend on which data we can actually obtain.

---

# 19. Training, Validation, and Testing

Once the dataset is ready, it should be separated for machine learning.

```text
Complete Dataset
       ↓
 ┌─────┼─────────┐
 ↓     ↓         ↓
Train Validation Test
```

## Training Data

Used to train the model.

## Validation Data

Used to tune and compare modelling choices.

## Test Data

Used to evaluate the final model using previously unseen data.

Because our data may be time-series data, the splitting strategy needs special attention. Randomly mixing future observations into the training data can create data leakage.

---

# 20. How Will We Evaluate the Model?

The prediction of energy consumption is a numerical prediction task, so it is a regression problem.

Possible evaluation metrics include:

## MAE

Mean Absolute Error:

```text
MAE = Average(|Actual - Predicted|)
```

It represents the average absolute prediction error.

---

## RMSE

Root Mean Squared Error:

```text
RMSE = √Average((Actual - Predicted)²)
```

It gives greater weight to larger errors.

---

## R²

R² measures how much of the variation in the target is explained by the model.

The final metrics should be selected according to the project's modelling requirements.

---

# 21. Complete Data Workflow

The complete process can be represented as:

```text
1. Define the problem
          ↓
2. Define the prediction target
          ↓
3. Identify required variables
          ↓
4. Classify the variables
          ↓
5. Identify possible data sources
          ↓
6. Collect the data
          ↓
7. Combine different datasets
          ↓
8. Check data quality
          ↓
9. Clean the data
          ↓
10. Standardize units and formats
          ↓
11. Handle missing values
          ↓
12. Check outliers and errors
          ↓
13. Create useful features
          ↓
14. Explore the dataset
          ↓
15. Select model inputs
          ↓
16. Define model output
          ↓
17. Prepare train/validation/test data
          ↓
18. Build the machine-learning model
          ↓
19. Evaluate the model
          ↓
20. Interpret the results
```

---

# 22. Final Checklist for the Data Phase

The original questions were:

1. What data will we need?
2. How can we process it?
3. What can we do with it?
4. How can we get it?
5. What type of data will we work with?

These should be expanded into the following complete checklist:

| # | Question | Purpose |
|---|---|---|
| 1 | What are we trying to predict? | Define the objective |
| 2 | What data do we need? | Identify variables |
| 3 | What is the target variable? | Define the prediction output |
| 4 | What are the input features? | Define model inputs |
| 5 | What types of data do we have? | Understand the data structure |
| 6 | Where can we get the data? | Identify sources |
| 7 | How can we collect the data? | Define acquisition methods |
| 8 | How can we combine the data? | Build one usable dataset |
| 9 | How can we clean the data? | Remove or correct data problems |
| 10 | How can we handle missing data? | Prevent incomplete data from causing problems |
| 11 | How can we check data quality? | Ensure reliability |
| 12 | How can we transform the data? | Make data suitable for modelling |
| 13 | What features can we create? | Extract useful information |
| 14 | What can we learn from the data? | Perform exploratory analysis |
| 15 | What can we predict with it? | Identify practical applications |
| 16 | How will we split the data? | Prepare for machine learning |
| 17 | How will we evaluate the model? | Measure prediction performance |
| 18 | What are the limitations of our data? | Understand reliability and scope |

---

# 23. Main Deliverable for This Week

The main deliverable should be a **Data Requirements Table**.

Recommended structure:

| Variable | Meaning | Unit | Data Type | Why Needed | Source | Collection Method | Format | Processing Required | Availability |
|---|---|---|---|---|---|---|---|---|---|
| Temperature | Ambient temperature | °C | Numerical | Environmental effect | Weather source | API/dataset | CSV/JSON | Cleaning | To verify |
| Speed | Vehicle speed | km/h | Numerical/time-series | Driving condition | Vehicle data | Measurement | CSV | Cleaning | To verify |
| Wind Speed | Wind velocity | m/s | Numerical | Aerodynamic effect | Weather source | API/dataset | CSV/JSON | Unit conversion | To verify |
| Elevation | Vehicle altitude | m | Numerical/time-series | Road gradient | Geographic source | GPS/API | CSV/JSON | Feature engineering | To verify |
| Traffic | Traffic condition | Category | Categorical | Driving behaviour | Traffic source | API/dataset | CSV/JSON | Encoding | To verify |
| Payload | Vehicle load | kg | Numerical | Vehicle mass | Measurement/assumption | Measurement | CSV | Validation | To verify |
| HVAC | Climate-control usage | On/Off or kW | Boolean/numerical | Electrical load | Vehicle data | Measurement | CSV | Encoding | To verify |
| Tire Pressure | Tire pressure | bar | Numerical | Rolling resistance | Vehicle data | Measurement | CSV | Validation | To verify |
| Consumption | Energy consumption | kWh/km | Numerical | **Prediction target** | Vehicle/experiment | Measurement/calculation | CSV | Calculation | To verify |

---

# 24. Final Structure of the Project's Data Phase

The data phase can therefore be summarized into six major questions:

```text
┌──────────────────────────────────────┐
│ 1. WHAT DATA DO WE NEED?             │
│                                      │
│ Vehicle + Environment + Road +       │
│ Traffic + Battery + Electrical Load  │
└──────────────────┬───────────────────┘
                   ↓
┌──────────────────────────────────────┐
│ 2. WHAT TYPE OF DATA IS IT?          │
│                                      │
│ Numerical + Categorical + Boolean +  │
│ Time-Series                          │
└──────────────────┬───────────────────┘
                   ↓
┌──────────────────────────────────────┐
│ 3. WHERE CAN WE GET IT?              │
│                                      │
│ Vehicle + Weather + Geographic +     │
│ Traffic + Technical Sources          │
└──────────────────┬───────────────────┘
                   ↓
┌──────────────────────────────────────┐
│ 4. HOW DO WE PROCESS IT?             │
│                                      │
│ Clean + Integrate + Transform +      │
│ Feature Engineering                  │
└──────────────────┬───────────────────┘
                   ↓
┌──────────────────────────────────────┐
│ 5. WHAT CAN WE DO WITH IT?           │
│                                      │
│ Analysis + Relationships + Prediction│
│ + Potential Range Estimation         │
└──────────────────┬───────────────────┘
                   ↓
┌──────────────────────────────────────┐
│ 6. IS THE DATA GOOD ENOUGH?          │
│                                      │
│ Quality + Completeness + Accuracy +  │
│ Consistency + Limitations            │
└──────────────────────────────────────┘
```

The final goal of this week's work is to move from **"What data do we need?"** to a clearly documented dataset specification that tells us **what each variable means, where it comes from, how it will be processed, and how it will eventually be used by the predictive model**.