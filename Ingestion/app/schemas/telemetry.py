"""Pydantic model = the telemetry contract (docs/telemetry_contract.md) in code.

Phase 3 enforces STRUCTURE only: required fields, types, timestamp format.
Physical ranges (GPS bounds, SoC 0-100, ...) are Phase 5, on purpose, so that
"malformed" (cannot be parsed) and "implausible" (parsed, but wrong) stay separate.
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, field_validator

# Strict: the JSON string "50" is NOT accepted as the number 50.
Number = Annotated[float, Field(strict=True)]


class Source(str, Enum):
    simulator = "simulator"
    volvo_api = "volvo_api"
    obd_ii = "obd_ii"
    can_logger = "can_logger"
    other_api = "other_api"


class ChargingState(str, Enum):
    driving = "driving"
    parked = "parked"
    charging = "charging"
    unknown = "unknown"


class TelemetryIn(BaseModel):
    model_config = ConfigDict(extra="forbid")  # unknown fields are rejected: catches typos

    # identity / metadata
    schema_version: str = Field(pattern=r"^1\.0$")
    vehicle_id: StrictStr = Field(min_length=1, max_length=64)
    trip_id: Optional[StrictStr] = Field(default=None, min_length=1, max_length=64)
    source: Source
    sequence_number: Optional[StrictInt] = None
    timestamp: datetime  # event time, UTC. received_at is added by the server.

    # position / motion
    latitude: Number
    longitude: Number
    altitude_m: Optional[Number] = None
    heading_deg: Optional[Number] = None
    speed_kmh: Number
    acceleration_mps2: Optional[Number] = None
    odometer_km: Optional[Number] = None

    # battery
    soc_pct: Number
    battery_voltage_v: Optional[Number] = None
    battery_current_a: Optional[Number] = None  # + discharge, - charge/regen
    battery_power_kw: Optional[Number] = None   # + discharge, - charge/regen
    battery_temp_c: Number
    charging_state: Optional[ChargingState] = None

    # climate / environment
    hvac_on: Optional[StrictBool] = None
    hvac_power_w: Optional[Number] = None
    hvac_setpoint_c: Optional[Number] = None
    cabin_temp_c: Optional[Number] = None
    ambient_temp_c: Optional[Number] = None

    # settings
    drive_mode: Optional[StrictStr] = None
    regen_level: Optional[StrictStr | StrictInt] = None

    @field_validator("timestamp", mode="before")
    @classmethod
    def timestamp_must_be_iso_string(cls, v: Any) -> Any:
        if not isinstance(v, str):
            raise ValueError("timestamp must be an ISO 8601 string, e.g. 2026-10-05T08:15:30.250Z")
        return v

    @field_validator("timestamp", mode="after")
    @classmethod
    def timestamp_must_have_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("timestamp must include a timezone (use 'Z' for UTC)")
        return v.astimezone(timezone.utc)


class TelemetryAccepted(BaseModel):
    status: str = "accepted"
    record_id: str
    received_at: datetime
    vehicle_id: str
