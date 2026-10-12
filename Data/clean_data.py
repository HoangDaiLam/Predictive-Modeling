import pandas as pd

NUMERIC_COLUMNS = [
    "ambient_temperature", "wind_speed", "elevation", "traffic_density",
    "total_payload", "tire_pressure", "battery_health", "battery_consumption",
]
CATEGORICAL_COLUMNS = ["weather", "performance_mode"]
REQUIRED_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def clean_data(file_path):
    df = pd.read_csv(file_path)

    # Standardize column names
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(r"\W+", "_", regex=True)
    )

    # Check required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Convert types (invalid numbers become NaN)
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("string").str.strip().str.lower()

    # Remove rows with missing values
    df = df.dropna(subset=REQUIRED_COLUMNS)

    # Remove rows with impossible values
    df = df[
        (df["wind_speed"] >= 0)
        & (df["total_payload"] >= 0)
        & (df["tire_pressure"] > 0)
        & (df["battery_health"].between(0, 100))
    ]

    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = clean_data("generated_data_test.csv")
    df.to_csv("cleaned_data.csv", index=False)
    print(f"Cleaned data saved: {len(df)} rows")