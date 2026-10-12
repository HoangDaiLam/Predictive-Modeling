from pathlib import Path
import argparse
import polars as pl
import numpy as np
import matplotlib.pyplot as plt


TARGET = "consumption_kwh_per_100km"
LEAKY = {"energy_used_kwh", "battery_level_end_pct", TARGET}
NUMERIC_COLUMNS = [
    "distance_km", "speed_kmh", "outside_temperature_c",
    "battery_temperature_c", "battery_level_start_pct",
    "battery_level_end_pct", "tyre_pressure_bar", "slope_grade_pct",
    "energy_used_kwh", "consumption_kwh_per_100km",
]
PREDICTOR_CANDIDATES = [c for c in NUMERIC_COLUMNS if c not in LEAKY]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/battery_data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/eda"))
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Could not find {args.input}. Run generate_battery_data.py first."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pl.read_csv(args.input)

    required = {TARGET, "speed_kmh", "outside_temperature_c", "energy_used_kwh",
                "distance_km", "observation_id", "vehicle_id", "trip_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {sorted(missing)}")

    print(f"Rows: {df.height:,} | Columns: {len(df.columns)}")
    print("\nSchema:")
    print(df.schema)
    print("\nSummary statistics:")
    print(df.select([c for c in NUMERIC_COLUMNS if c in df.columns]).describe())

    d = df.select(((pl.col("energy_used_kwh") / pl.col("distance_km") * 100) - pl.col(TARGET)).abs().max()).item()
    print(f"\nmax |energy/dist*100 - target|: {d:.5f}")
    print("duplicate rows:", int(df.is_duplicated().sum()),
          "| duplicate observation_id:", df.height - df["observation_id"].n_unique())
    print("vehicles:", df["vehicle_id"].n_unique(), "| trips:", df["trip_id"].n_unique())

    # 1. Target distribution
    values = df["consumption_kwh_per_100km"].drop_nulls().to_numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=35, edgecolor="black")
    ax.set(title="Distribution of energy consumption",
           xlabel="Consumption (kWh / 100 km)", ylabel="Number of observations")
    fig.tight_layout()
    fig.savefig(args.output_dir / "01_consumption_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Consumption vs speed
    x = df["speed_kmh"].to_numpy()
    y = df["consumption_kwh_per_100km"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, alpha=0.28, s=12)
    ax.set(title="Speed vs energy consumption", xlabel="Speed (km/h)",
           ylabel="Consumption (kWh / 100 km)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "02_speed_vs_consumption.png", dpi=150)
    plt.close(fig)

    # 3. Consumption vs outside temperature
    x = df["outside_temperature_c"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, alpha=0.28, s=12)
    ax.set(title="Outside temperature vs energy consumption",
           xlabel="Outside temperature (°C)", ylabel="Consumption (kWh / 100 km)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "03_temperature_vs_consumption.png", dpi=150)
    plt.close(fig)

    # 5. Correlation matrix for numeric columns
    numeric = [c for c in PREDICTOR_CANDIDATES + [TARGET] if c in df.columns]
    corr = df.select(numeric).drop_nulls().corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(numeric)), labels=numeric, rotation=75, ha="right")
    ax.set_yticks(range(len(numeric)), labels=numeric)
    ax.set_title("Correlation matrix (numeric variables)")
    fig.colorbar(im, ax=ax, label="Correlation")
    fig.tight_layout()
    fig.savefig(args.output_dir / "05_correlation_matrix.png", dpi=150)
    plt.close(fig)

    def box_by(col, order, fname):
        if col not in df.columns:
            return
        labels, groups = [], []
        for name in order:
            arr = df.filter(pl.col(col) == name)[TARGET].drop_nulls().to_numpy()
            if len(arr):
                labels.append(name); groups.append(arr)
        if groups:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.boxplot(groups)
            ax.set_xticklabels(labels)
            ax.set(title=f"Consumption by {col}", xlabel=col, ylabel="Consumption (kWh / 100 km)")
            fig.tight_layout(); fig.savefig(args.output_dir / fname, dpi=150); plt.close(fig)

    box_by("traffic_situation", ["low", "medium", "high"], "04_traffic_boxplot.png")
    box_by("road_condition", ["dry", "wet", "snow_or_ice"], "06_road_condition_boxplot.png")
    box_by("climate_settings", ["off", "low", "medium", "high"], "07_climate_boxplot.png")
    box_by("preconditioning", ["off", "on"], "08_preconditioning_boxplot.png")
    box_by("towing", ["no", "yes"], "09_towing_boxplot.png")
    box_by("current_driving_pattern", ["steady", "mixed", "stop_start"], "10_pattern_boxplot.png")
    box_by("historical_driving_pattern", ["efficient", "typical", "energy_intensive"], "11_hist_pattern_boxplot.png")

    # Basic quality checks
    print("\nBasic data-quality checks:")
    print("Null counts:")
    print(df.null_count())
    print("\nRows with negative energy used:",
          df.filter(pl.col("energy_used_kwh") < 0).height)
    print("\nPlots saved to:", args.output_dir.resolve())


if __name__ == "__main__":
    main()