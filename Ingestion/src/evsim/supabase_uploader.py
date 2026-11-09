from __future__ import annotations

from supabase import Client


def _insert_batches(
    client: Client,
    table: str,
    rows: list[dict],
    batch_size: int,
    on_conflict: str | None = None,
) -> None:
    if not rows:
        return

    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        query = client.table(table).upsert(
            batch,
            on_conflict=on_conflict
        ) if on_conflict else client.table(table).insert(batch)
        query.execute()
        print(f"{table}: uploaded {min(offset + len(batch), len(rows))}/{len(rows)}")


def seed_supabase(
    client: Client,
    dataset,
    batch_size: int = 500,
) -> str:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    # Upsert run by unique dataset_version, then retrieve its generated UUID.
    run_response = (
        client.table("dataset_runs")
        .upsert(dataset.dataset_run, on_conflict="dataset_version")
        .execute()
    )
    if not run_response.data:
        raise RuntimeError("Could not create or retrieve dataset run")
    dataset_id = run_response.data[0]["dataset_id"]

    # Attach dataset_id to every record before upload.
    for collection in (
        dataset.vehicles, dataset.routes, dataset.trips,
        dataset.telemetry, dataset.trip_results
    ):
        for row in collection:
            row["dataset_id"] = dataset_id

    # Parent records must be uploaded before child records.
    _insert_batches(
        client, "vehicles", dataset.vehicles, batch_size,
        on_conflict="dataset_id,vehicle_id"
    )
    _insert_batches(
        client, "routes", dataset.routes, batch_size,
        on_conflict="dataset_id,route_id"
    )
    _insert_batches(
        client, "trips", dataset.trips, batch_size,
        on_conflict="dataset_id,trip_id"
    )
    _insert_batches(
        client, "telemetry", dataset.telemetry, batch_size,
        on_conflict="dataset_id,trip_id,timestamp"
    )
    _insert_batches(
        client, "trip_results", dataset.trip_results, batch_size,
        on_conflict="dataset_id,trip_id"
    )

    # Update dataset-run counts after successful uploads.
    client.table("dataset_runs").update({
        "vehicle_count": len(dataset.vehicles),
        "route_count": len(dataset.routes),
        "trip_count": len(dataset.trips),
        "telemetry_count": len(dataset.telemetry),
    }).eq("dataset_id", dataset_id).execute()

    return dataset_id
