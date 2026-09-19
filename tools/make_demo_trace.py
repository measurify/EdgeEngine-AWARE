#!/usr/bin/env python3
"""Generate a *synthetic* one-year hourly trace in the same format as
``tools/fetch_open_meteo.py`` so that the trace-driven backends, tests and
notebook run without network access.

THIS IS NOT MEASURED DATA. It is a physically-motivated stand-in for a
Ligurian coastal horticultural site (44.05 N, 8.21 E): astronomical clear-sky
irradiance (Haurwitz), persistent cloudiness, Markov wet/dry days with gamma
rainfall, a single-layer soil-water bucket with drainage, evapotranspiration
and summer irrigation, and a seasonal/diurnal temperature model. The
generator is deliberately *different* from the simulator's stochastic models
(different cloud process, seasonal cycle, a real soil-water bucket), so a
policy trained on the synthetic environment meets genuinely new dynamics
when evaluated on it. Replace it with the real archive as soon as you can.

    python tools/make_demo_trace.py --out data/traces/demo_liguria_2023_hourly.csv
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from pathlib import Path

import numpy as np

VARIABLES = ["shortwave_radiation", "soil_moisture_0_to_7cm", "temperature_2m", "relative_humidity_2m", "precipitation"]
KINDS = {"shortwave_radiation": "preceding_mean", "precipitation": "preceding_mean"}
UNITS = {"shortwave_radiation": "W/m²", "soil_moisture_0_to_7cm": "m³/m³", "temperature_2m": "°C", "relative_humidity_2m": "%", "precipitation": "mm"}


def clear_sky_ghi(lat_deg: float, lon_deg: float, doy: int, hour_local: float, utc_offset_h: float) -> float:
    """Haurwitz clear-sky GHI [W/m²] for a given day of year and local standard hour."""
    decl = math.radians(23.45) * math.sin(2 * math.pi * (284 + doy) / 365.0)
    b = 2 * math.pi * (doy - 81) / 364.0
    eot_min = 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)  # equation of time
    solar_hour = hour_local + (lon_deg - 15.0 * utc_offset_h) / 15.0 + eot_min / 60.0
    ha = math.radians(15.0 * (solar_hour - 12.0))
    lat = math.radians(lat_deg)
    cos_z = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(ha)
    if cos_z <= 0.0:
        return 0.0
    return 1098.0 * cos_z * math.exp(-0.059 / cos_z)


def generate(year: int, lat: float, lon: float, utc_offset_h: float, seed: int) -> tuple[list[dt.datetime], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    t0 = dt.datetime(year, 1, 1)
    n_days = (dt.datetime(year + 1, 1, 1) - t0).days
    n = n_days * 24 + 1  # include the closing midnight

    # ---- daily weather regime: clearness and wet/dry Markov chain ----------
    k_day = np.zeros(n_days)
    wet = np.zeros(n_days, dtype=bool)
    rain_day_mm = np.zeros(n_days)
    k = 0.7
    prev_wet = False
    for d in range(n_days):
        season = math.cos(2 * math.pi * (d - 196) / 365.0)  # +1 mid July, -1 mid January
        p_wet = 0.09 + 0.12 * (1 - season) / 2 + (0.25 if prev_wet else 0.0)
        prev_wet = wet[d] = rng.random() < p_wet
        k_mean = 0.66 + 0.12 * season
        k = k_mean + 0.55 * (k - k_mean) + rng.normal(0, 0.16)
        if wet[d]:
            k = min(k, rng.uniform(0.15, 0.45))
        k_day[d] = float(np.clip(k, 0.08, 0.98))
        if wet[d]:
            rain_day_mm[d] = rng.gamma(shape=1.4, scale=6.0 + 4.0 * (1 - season) / 2)  # ~10 mm per wet day

    # ---- hourly series ------------------------------------------------------
    times = [t0 + dt.timedelta(hours=h) for h in range(n)]
    ghi = np.zeros(n)
    precip = np.zeros(n)
    temp = np.zeros(n)
    rh = np.zeros(n)
    theta = np.zeros(n)

    # hours of each wet day that receive rain, and their share of the daily amount
    rain_hours: dict[int, float] = {}
    for d in range(n_days):
        if wet[d] and rain_day_mm[d] > 0:
            n_h = int(rng.integers(2, 8))
            hours = rng.choice(24, size=n_h, replace=False)
            shares = rng.dirichlet(np.ones(n_h))
            for h, sh in zip(hours, shares):
                rain_hours[d * 24 + int(h) + 1] = rain_day_mm[d] * float(sh)  # stamped at the end of the hour

    cloud_dev = 0.0
    temp_dev = 0.0
    fc, wp, depth_mm = 0.40, 0.10, 70.0  # field capacity, wilting point, layer depth (0-7 cm)
    th = 0.30
    irrigation_timer = -1
    for i, t in enumerate(times):
        d = min(i // 24, n_days - 1)
        doy = t.timetuple().tm_yday
        season = math.cos(2 * math.pi * (doy - 197) / 365.0)
        hour_mid = t.hour - 0.5  # preceding-hour mean → evaluate at the middle of the hour
        # irradiance (mean of the preceding hour, as in the archive)
        cloud_dev = 0.75 * cloud_dev + rng.normal(0, 0.10)
        kt = float(np.clip(k_day[d] + cloud_dev, 0.05, 1.0))
        ghi[i] = clear_sky_ghi(lat, lon, doy, hour_mid, utc_offset_h) * kt if i > 0 else 0.0
        # precipitation: the day's amount spread over a few hours (see above)
        precip[i] = rain_hours.get(i, 0.0)
        # temperature: seasonal mean, diurnal amplitude damped by clouds, AR(1) noise
        t_mean = 15.5 + 8.5 * season
        amp = 4.0 + 4.0 * kt
        temp_dev = 0.9 * temp_dev + rng.normal(0, 0.5)
        temp[i] = t_mean + amp * math.cos(2 * math.pi * (t.hour - 15) / 24.0) + temp_dev - (2.0 if precip[i] > 0 else 0.0)
        # humidity anti-correlated with temperature anomaly, wet when raining
        rh[i] = float(np.clip(68 - 2.2 * (temp[i] - t_mean) + (18 if precip[i] > 0 else 0) + rng.normal(0, 3), 20, 100))
        # soil water bucket (top 7 cm, mm) -----------------------------------
        et0_day = 1.0 + 3.6 * (1 + season) / 2  # mm/day reference ET, ~1 (winter) .. ~4.6 (summer)
        et_hour = et0_day / 24.0 * (0.3 + 2.4 * (ghi[i] / 700.0)) * ((0.3 + 0.7 * min(1.0, (th - wp) / (fc - wp))) if th > wp else 0.0)
        drain = max(0.0, th - fc) * depth_mm * 0.35  # fast drainage of water above field capacity
        irrig = 0.0
        if season > 0.2:  # growing season: a farmer irrigates when the soil dries out
            if th < 0.11 and irrigation_timer < 0:  # below the crop's stress level
                irrigation_timer = int(rng.integers(6, 36))
            if irrigation_timer == 0:
                irrig = 12.0
            irrigation_timer -= 1 if irrigation_timer >= 0 else 0
        th = th + (precip[i] + irrig - et_hour - drain) / depth_mm
        th = float(np.clip(th, wp * 0.8, 0.48))
        theta[i] = th + rng.normal(0, 0.0015)

    return times, {
        "shortwave_radiation": np.round(ghi, 1),
        "soil_moisture_0_to_7cm": np.round(theta, 4),
        "temperature_2m": np.round(temp, 2),
        "relative_humidity_2m": np.round(rh, 1),
        "precipitation": np.round(precip, 2),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", type=int, default=2023)
    p.add_argument("--lat", type=float, default=44.05)
    p.add_argument("--lon", type=float, default=8.21)
    p.add_argument("--utc-offset", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--out", type=Path, default=Path("data/traces/demo_liguria_2023_hourly.csv"))
    args = p.parse_args(argv)

    times, cols = generate(args.year, args.lat, args.lon, args.utc_offset, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        fh.write("# site=SYNTHETIC demo trace, Liguria-like coastal site (NOT measured data)\n")
        fh.write(f"# latitude={args.lat}\n# longitude={args.lon}\n")
        fh.write(f"# source=tools/make_demo_trace.py seed={args.seed} (stand-in for the Open-Meteo archive)\n")
        fh.write(f"# utc_offset_h={args.utc_offset}\n")
        fh.write(f"# start_local={times[0].isoformat(timespec='minutes')}\n")
        for v in VARIABLES:
            fh.write(f"# unit:{v}={UNITS[v]}\n")
            fh.write(f"# kind:{v}={KINDS.get(v, 'instant')}\n")
        fh.write("time," + ",".join(VARIABLES) + "\n")
        for i, t in enumerate(times):
            fh.write(t.isoformat(timespec="minutes") + "," + ",".join(f"{cols[v][i]:g}" for v in VARIABLES) + "\n")
    print(f"wrote {len(times)} rows to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
