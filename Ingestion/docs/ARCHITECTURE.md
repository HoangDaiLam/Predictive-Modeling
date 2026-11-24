# Architecture

1. `generator.py` creates deterministic synthetic vehicles, routes, trips,
   telemetry and trip results.
2. `validation.py` checks referential consistency and basic physical ranges.
3. `supabase_uploader.py` uploads data in batches, respecting foreign-key order.
4. Supabase PostgreSQL stores the normalized tables.
5. `ml_dataset` is a database view with one row per trip.
6. Teammates read the view through Supabase's REST API using a publishable key.

No separate FastAPI server is needed for this first iteration.

## Security boundary

The seed script uses a Supabase secret key and is trusted backend code.
Teammates use the publishable key and read-only RLS policies. Never share the
secret key. Public read access is only appropriate because this database
contains synthetic, non-sensitive data.
