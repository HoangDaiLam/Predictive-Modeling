import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ev_platform.generator import generate_dataset
from ev_platform.quality import quality_report, write_quality_report


def main():
    parser = argparse.ArgumentParser(description="Generate and validate synthetic EV trip data.")
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--output", default="data/ev_dataset.csv")
    args = parser.parse_args()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(args.rows, args.seed, args.version)
    report = quality_report(df)
    if not report["passed"]:
        raise SystemExit(f"Quality gate failed: {report}")
    df.to_csv(output, index=False, float_format="%.6f")
    report_path = ROOT / "data" / "reports" / f"quality_{args.version}.json"
    write_quality_report(df, report_path)
    print(f"Generated {len(df):,} trips: {output}")
    print(f"Quality checks passed: {report_path}")
    print(f"Mean consumption: {df.consumption_kwh_per_100km.mean():.2f} kWh/100 km")

if __name__ == "__main__":
    main()
