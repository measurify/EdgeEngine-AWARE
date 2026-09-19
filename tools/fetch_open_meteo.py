#!/usr/bin/env python3
"""Download an hourly weather / soil trace from the Open-Meteo historical
archive (ERA5 / ERA5-Land reanalysis) in the CSV format read by
``edgeengine_aware.traces.Trace.from_csv``.

Example (Albenga, Liguria - a coastal horticultural plain):

    python tools/fetch_open_meteo.py --lat 44.05 --lon 8.21 \
        --start 2023-01-01 --end 2023-12-31 \
        --out data/traces/albenga_2023_hourly.csv

Variables (hourly):
    shortwave_radiation    W/m²   global horizontal irradiance, mean of the preceding hour
    soil_moisture_0_to_7cm m³/m³  volumetric water content of the top soil layer
    temperature_2m         °C
    relative_humidity_2m   %
    precipitation          mm     sum of the preceding hour

Timestamps are converted to *local standard time* (fixed UTC offset, no
daylight-saving jumps) so that every day has exactly 24 rows and the
simulator's time of day matches the sun. The first rows before the first
local midnight are dropped.

Data: © Open-Meteo.com, CC BY 4.0 (https://open-meteo.com/en/terms);
reanalysis by ECMWF (ERA5 / ERA5-Land, Copernicus Climate Change Service).
Only the standard library is required.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

VARIABLES = ["shortwave_radiation", "soil_moisture_0_to_7cm", "temperature_2m", "relative_humidity_2m", "precipitation"]
KINDS = {
    "shortwave_radiation": "preceding_mean",
    "precipitation": "preceding_mean",
    "soil_moisture_0_to_7cm": "instant",
    "temperature_2m": "instant",
    "relative_humidity_2m": "instant",
}
API = "https://archive-api.open-meteo.com/v1/archive"


def fetch(lat: float, lon: float, start: str, end: str, variables: list[str]) -> dict:
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(variables),
        "timezone": "UTC",
    }
    url = API + "?" + urllib.parse.urlencode(params)
    print("GET", url, file=sys.stderr)
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 - fixed https host
        return json.load(resp)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lat", type=float, default=44.05)
    p.add_argument("--lon", type=float, default=8.21)
    p.add_argument("--start", default="2023-01-01", help="first local day (YYYY-MM-DD)")
    p.add_argument("--end", default="2023-12-31", help="last local day (YYYY-MM-DD)")
    p.add_argument("--utc-offset", type=float, default=1.0, help="local standard time offset in hours (Italy: 1)")
    p.add_argument("--site", default="Albenga, Liguria, IT")
    p.add_argument("--out", type=Path, default=Path("data/traces/albenga_2023_hourly.csv"))
    p.add_argument("--json", type=Path, default=None, help="also keep the raw API response")
    args = p.parse_args(argv)

    start_d = dt.date.fromisoformat(args.start)
    end_d = dt.date.fromisoformat(args.end)
    # fetch one UTC day of margin on both sides so that the local window is complete
    data = fetch(args.lat, args.lon, (start_d - dt.timedelta(days=1)).isoformat(), (end_d + dt.timedelta(days=1)).isoformat(), VARIABLES)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data))
    hourly = data["hourly"]
    times_utc = [dt.datetime.fromisoformat(t) for t in hourly["time"]]
    offset = dt.timedelta(hours=args.utc_offset)
    local_start = dt.datetime.combine(start_d, dt.time())
    local_end = dt.datetime.combine(end_d + dt.timedelta(days=1), dt.time())  # inclusive last midnight

    rows = []
    for i, t in enumerate(times_utc):
        tl = t + offset
        if not local_start <= tl <= local_end:
            continue
        vals = [hourly[v][i] for v in VARIABLES]
        rows.append((tl, vals))
    if not rows:
        print("no data in the requested window", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        fh.write(f"# site={args.site}\n")
        fh.write(f"# latitude={args.lat}\n# longitude={args.lon}\n")
        fh.write(f"# elevation_m={data.get('elevation', '')}\n")
        fh.write("# source=Open-Meteo historical weather API (ERA5 / ERA5-Land reanalysis), CC BY 4.0\n")
        fh.write(f"# utc_offset_h={args.utc_offset}\n")
        fh.write(f"# start_local={rows[0][0].isoformat(timespec='minutes')}\n")
        fh.write(f"# fetched={dt.datetime.now(dt.timezone.utc).isoformat(timespec='minutes')}\n")
        units = data.get("hourly_units", {})
        for v in VARIABLES:
            fh.write(f"# unit:{v}={units.get(v, '')}\n")
            fh.write(f"# kind:{v}={KINDS[v]}\n")
        fh.write("time," + ",".join(VARIABLES) + "\n")
        for tl, vals in rows:
            fh.write(tl.isoformat(timespec="minutes") + "," + ",".join("" if v is None else f"{v:g}" for v in vals) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
