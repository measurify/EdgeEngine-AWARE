"""Trace-driven backends: replay recorded weather / soil data instead of the
stochastic models.

The stochastic models in ``energy.py`` and ``agriculture.py`` are convenient
for training (unlimited, controllable variety) but they are *models*. This
module lets the same environment run on **recorded traces** - e.g. hourly
irradiance and soil moisture of a real site from an open weather archive
(``tools/fetch_open_meteo.py``) or from the user's own field loggers - so that
a policy trained on the synthetic models can be evaluated on data the models
never saw (a first sim-to-real check), or trained on real seasons directly.

Design
------
* :class:`Trace` - a time-indexed table. Each column is either an *instant*
  series (soil moisture, temperature: linearly interpolated) or a
  *preceding-interval mean* (irradiance, precipitation as delivered by weather
  archives: piecewise constant over the interval that ends at the timestamp).
  Time is in seconds since the trace's first local midnight, so the
  environment's time-of-day matches the trace's.
* :class:`TraceSolarEnergySource` - drop-in replacement of
  ``energy.SolarEnergySource``: panel power = ``max_power_w * efficiency *
  GHI / 1000 W/m²``, integrated over the decision interval.
* :class:`TraceFieldEnvironment` - drop-in replacement of
  ``agriculture.FieldEnvironment``: replays soil moisture (converted from
  volumetric content to *fraction of field capacity*), temperature and
  humidity; rain events are derived from the precipitation column.
* :class:`TraceDrivenEnv` - ``EdgeEngineAwareEnv`` whose harvesting and/or
  field backends are trace-driven. Every episode replays a window of the trace
  starting at a (random or fixed) day; everything else (battery, radio,
  application, reward, observation contract) is unchanged, so **the policy
  cannot tell the two worlds apart** - which is exactly the point.

The node still only sees measurable quantities: the trace feeds the *ground
truth*; the sensor and the harvester monitor add their noise as before.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .agriculture import FieldState
from .config import AgricultureConfig, EdgeEngineAwareConfig, HarvestingConfig
from .env import EdgeEngineAwareEnv

DAY_S = 86400.0
HOUR_S = 3600.0

# Column kinds: how a value at timestamp t is to be interpreted.
INSTANT = "instant"  # sampled at t; linear interpolation between samples
PRECEDING_MEAN = "preceding_mean"  # mean over (t - step, t]; piecewise constant

#: Default interpretation of the Open-Meteo / ERA5-Land hourly variables.
DEFAULT_KINDS: dict[str, str] = {
    "shortwave_radiation": PRECEDING_MEAN,  # W/m², mean of the preceding hour
    "precipitation": PRECEDING_MEAN,  # mm, sum of the preceding hour
    "soil_moisture_0_to_7cm": INSTANT,  # m³/m³
    "temperature_2m": INSTANT,  # °C
    "relative_humidity_2m": INSTANT,  # %
}


# ---------------------------------------------------------------------------
# Trace container
# ---------------------------------------------------------------------------
@dataclass
class TraceColumn:
    values: np.ndarray
    kind: str = INSTANT
    unit: str = ""


@dataclass
class Trace:
    """A time-indexed table of recorded quantities.

    ``time_s`` is strictly increasing, in seconds since the first local
    midnight of the recording (``time_s[0]`` may be > 0 if the recording does
    not start at midnight). ``meta`` carries provenance (site, source, UTC
    offset, ...) and is written back as ``# key=value`` header lines by
    :meth:`to_csv`.
    """

    time_s: np.ndarray
    columns: dict[str, TraceColumn]
    meta: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time_s = np.asarray(self.time_s, dtype=float)
        if self.time_s.ndim != 1 or len(self.time_s) < 2:
            raise ValueError("a trace needs at least two samples")
        if np.any(np.diff(self.time_s) <= 0):
            raise ValueError("trace timestamps must be strictly increasing")
        for name, col in self.columns.items():
            col.values = np.asarray(col.values, dtype=float)
            if col.values.shape != self.time_s.shape:
                raise ValueError(f"column {name!r} has {col.values.shape[0]} samples, expected {len(self.time_s)}")
            if col.kind not in (INSTANT, PRECEDING_MEAN):
                raise ValueError(f"unknown column kind {col.kind!r} for {name!r}")

    # -- geometry -------------------------------------------------------------
    @property
    def start_s(self) -> float:
        return float(self.time_s[0])

    @property
    def end_s(self) -> float:
        return float(self.time_s[-1])

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def n_days(self) -> int:
        """Number of complete days available from the first midnight."""
        return int(self.end_s // DAY_S)

    @property
    def step_s(self) -> float:
        """Median sampling interval."""
        return float(np.median(np.diff(self.time_s)))

    def __contains__(self, name: str) -> bool:
        return name in self.columns

    def __len__(self) -> int:
        return len(self.time_s)

    # -- access ---------------------------------------------------------------
    def value_at(self, name: str, t: float | np.ndarray) -> float | np.ndarray:
        """Value of ``name`` at time ``t`` (clamped to the recording range).

        Instant columns are linearly interpolated; preceding-mean columns are
        piecewise constant: the value stamped ``t_k`` holds on ``(t_{k-1}, t_k]``.
        """
        col = self.columns[name]
        t_arr = np.clip(np.asarray(t, dtype=float), self.time_s[0], self.time_s[-1])
        if col.kind == INSTANT:
            out = np.interp(t_arr, self.time_s, col.values)
        else:
            idx = np.searchsorted(self.time_s, t_arr, side="left")
            idx = np.clip(idx, 0, len(self.time_s) - 1)
            out = col.values[idx]
        return float(out) if np.ndim(t) == 0 else out

    def interval_mean(self, name: str, t0: float, t1: float) -> float:
        """Time average of ``name`` over ``[t0, t1]`` (exact for both kinds)."""
        if t1 <= t0:
            return self.value_at(name, t0)  # type: ignore[return-value]
        lo, hi = self.time_s[0], self.time_s[-1]
        a, b = float(np.clip(t0, lo, hi)), float(np.clip(t1, lo, hi))
        if b <= a:  # entirely outside the recording: hold the boundary value
            return self.value_at(name, a)  # type: ignore[return-value]
        col = self.columns[name]
        inner = self.time_s[(self.time_s > a) & (self.time_s < b)]
        knots = np.concatenate(([a], inner, [b]))
        if col.kind == INSTANT:
            vals = np.interp(knots, self.time_s, col.values)
            integral = float(np.sum(0.5 * (vals[1:] + vals[:-1]) * np.diff(knots)))
        else:
            # piecewise constant on (t_{k-1}, t_k]: evaluate just right of each left knot
            mids = 0.5 * (knots[1:] + knots[:-1])
            vals = np.asarray(self.value_at(name, mids), dtype=float)
            integral = float(np.sum(vals * np.diff(knots)))
        return integral / (b - a)

    def interval_sum(self, name: str, t0: float, t1: float) -> float:
        """Integral of a *rate* column over ``[t0, t1]`` expressed in the
        column's per-sample units (e.g. mm for hourly precipitation sums)."""
        col = self.columns[name]
        if col.kind != PRECEDING_MEAN:
            raise ValueError("interval_sum is defined for preceding-mean columns only")
        return self.interval_mean(name, t0, t1) * (t1 - t0) / self.step_s

    def daily_means(self, name: str) -> np.ndarray:
        """Mean of ``name`` over each complete day ``[k*86400, (k+1)*86400)``."""
        return np.array([self.interval_mean(name, k * DAY_S, (k + 1) * DAY_S) for k in range(self.n_days)])

    def slice_days(self, start_day: int, n_days: int) -> "Trace":
        """Sub-trace covering days ``[start_day, start_day + n_days]`` re-based to day 0."""
        t0, t1 = start_day * DAY_S, (start_day + n_days) * DAY_S
        sel = (self.time_s >= t0) & (self.time_s <= t1)
        if sel.sum() < 2:
            raise ValueError("requested day range is outside the trace")
        cols = {n: TraceColumn(c.values[sel].copy(), c.kind, c.unit) for n, c in self.columns.items()}
        meta = dict(self.meta)
        meta["sliced_from_day"] = str(start_day)
        return Trace(self.time_s[sel] - t0, cols, meta)

    # -- I/O ------------------------------------------------------------------
    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        time_column: str = "time",
        kinds: Mapping[str, str] | None = None,
        columns: Iterable[str] | None = None,
    ) -> "Trace":
        """Load a trace from CSV.

        The file may start with ``# key=value`` metadata lines. ``time`` is
        either an ISO-8601 local timestamp (``2023-06-01T13:00``) or a number
        of seconds. Column kinds default to :data:`DEFAULT_KINDS` (instant for
        unknown names) and can be overridden with ``kinds``.
        """
        path = Path(path)
        meta: dict[str, str] = {}
        rows: list[dict[str, str]] = []
        with path.open(newline="") as fh:
            header_lines = []
            for line in fh:
                if line.startswith("#"):
                    body = line[1:].strip()
                    if "=" in body:
                        k, v = body.split("=", 1)
                        meta[k.strip()] = v.strip()
                    continue
                header_lines.append(line)
                break
            reader = csv.DictReader(header_lines + list(fh))
            for row in reader:
                if row.get(time_column) in (None, ""):
                    continue
                rows.append(row)
        if not rows:
            raise ValueError(f"{path} contains no data rows")
        names = [n for n in rows[0].keys() if n != time_column and n is not None]
        if columns is not None:
            names = [n for n in names if n in set(columns)]
        time_raw = [r[time_column] for r in rows]
        time_s = _parse_time(time_raw)
        kinds_map = dict(DEFAULT_KINDS)
        # ``# kind:<column>=<kind>`` header lines written by :meth:`to_csv`
        for k in [k for k in meta if k.startswith("kind:")]:
            kinds_map[k[len("kind:"):]] = meta.pop(k)
        kinds_map.update(kinds or {})
        cols = {}
        for n in names:
            vals = np.array([float(r[n]) if r[n] not in ("", None) else np.nan for r in rows], dtype=float)
            if np.isnan(vals).any():  # fill gaps by interpolation (archives have a few)
                ok = ~np.isnan(vals)
                if ok.sum() < 2:
                    raise ValueError(f"column {n!r} has no usable data")
                vals = np.interp(time_s, time_s[ok], vals[ok])
            cols[n] = TraceColumn(vals, kinds_map.get(n, INSTANT))
        meta.setdefault("file", path.name)
        return cls(time_s, cols, meta)

    def to_csv(self, path: str | Path, *, time_format: str = "seconds") -> None:
        """Write the trace (metadata as ``# key=value`` header lines)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            for k, v in self.meta.items():
                fh.write(f"# {k}={v}\n")
            for n, c in self.columns.items():
                fh.write(f"# kind:{n}={c.kind}\n")
            w = csv.writer(fh)
            w.writerow(["time", *self.columns])
            start = self.meta.get("start_local")
            for i, t in enumerate(self.time_s):
                if time_format == "iso" and start:
                    ts = (np.datetime64(start) + np.timedelta64(int(round(t)), "s")).astype("datetime64[m]")
                    tcol: Any = str(ts)
                else:
                    tcol = f"{t:.0f}"
                w.writerow([tcol, *(f"{c.values[i]:.6g}" for c in self.columns.values())])

    def describe(self) -> str:
        lines = [f"Trace: {len(self)} samples, {self.n_days} days, step {self.step_s/3600:.2f} h"]
        for k, v in self.meta.items():
            lines.append(f"  {k} = {v}")
        for n, c in self.columns.items():
            lines.append(f"  {n:28s} [{c.kind:14s}] min {c.values.min():9.3f}  mean {c.values.mean():9.3f}  max {c.values.max():9.3f}")
        return "\n".join(lines)


def _parse_time(raw: list[str]) -> np.ndarray:
    """ISO timestamps → seconds since the first local midnight; numbers → as is."""
    first = raw[0].strip()
    try:
        float(first)
        return np.array([float(x) for x in raw], dtype=float)
    except ValueError:
        pass
    ts = np.array([np.datetime64(x.strip()) for x in raw]).astype("datetime64[s]")
    midnight = ts[0].astype("datetime64[D]").astype("datetime64[s]")
    return (ts - midnight).astype("timedelta64[s]").astype(float)


# ---------------------------------------------------------------------------
# Episode window shared by all trace-driven backends of one environment
# ---------------------------------------------------------------------------
class TraceWindow:
    """Which part of the trace the current episode replays.

    ``offset_s`` is added to the environment clock (which starts at day 0)
    to index the trace; it is always a whole number of days so that the
    time of day is preserved.
    """

    def __init__(self, trace: Trace, episode_s: float, start_day: int | None = None):
        self.trace = trace
        self.episode_s = episode_s
        self.start_day = start_day
        # one extra day is needed because the harvesting source is started one
        # interval before the episode and the field looks one step ahead
        self.max_start_day = int(math.floor((trace.end_s - episode_s) / DAY_S)) - 1
        if self.max_start_day < 0:
            raise ValueError(
                f"trace covers {trace.end_s/DAY_S:.1f} days, episodes need {episode_s/DAY_S:.1f} (+1) days"
            )
        if start_day is not None and not 0 <= start_day <= self.max_start_day:
            raise ValueError(f"start_day must be in [0, {self.max_start_day}], got {start_day}")
        self.current_day = start_day if start_day is not None else 0

    def choose(self, rng: np.random.Generator, start_day: int | None = None) -> int:
        if start_day is None:
            start_day = self.start_day
        if start_day is None:
            start_day = int(rng.integers(0, self.max_start_day + 1))
        elif not 0 <= start_day <= self.max_start_day:
            raise ValueError(f"start_day must be in [0, {self.max_start_day}], got {start_day}")
        self.current_day = int(start_day)
        return self.current_day

    @property
    def offset_s(self) -> float:
        return self.current_day * DAY_S


# ---------------------------------------------------------------------------
# Harvesting backend
# ---------------------------------------------------------------------------
class TraceSolarEnergySource:
    """``SolarEnergySource`` twin driven by a recorded irradiance series.

    P(t) = max_power_w * efficiency * GHI(t) / reference_irradiance, averaged
    over the decision interval. ``max_power_w`` keeps its meaning of *panel
    output at the reference irradiance* (1000 W/m² = STC by default), so the
    hardware profile and the domain randomisation of ``HarvestingConfig`` apply
    unchanged; the cloud/clearness parameters are ignored (the trace *is* the
    weather).
    """

    def __init__(
        self,
        trace: Trace,
        cfg: HarvestingConfig,
        timestep_s: float,
        window: TraceWindow,
        *,
        column: str = "shortwave_radiation",
        reference_irradiance_w_m2: float = 1000.0,
    ):
        if column not in trace:
            raise KeyError(f"trace has no column {column!r}")
        self.trace = trace
        self.cfg = cfg
        self.dt = timestep_s
        self.window = window
        self.column = column
        self.ref_irr = reference_irradiance_w_m2
        self._rng = np.random.default_rng()
        self._daily_irr = trace.daily_means(column)
        self._daily_irr_max = float(self._daily_irr.max()) if len(self._daily_irr) else 1.0
        self._power_w = 0.0
        self._measured_w = 0.0
        self._time_s = 0.0

    def reset(self, rng: np.random.Generator, start_time_s: float) -> None:
        self._rng = rng
        self.update(start_time_s)

    def irradiance_w_m2(self, time_s: float) -> float:
        """Mean GHI over the interval starting at environment time ``time_s``."""
        t0 = time_s + self.window.offset_s
        return self.trace.interval_mean(self.column, t0, t0 + self.dt)

    def update(self, time_s: float) -> None:
        self._time_s = time_s
        c = self.cfg
        self._power_w = c.max_power_w * c.efficiency * max(0.0, self.irradiance_w_m2(time_s)) / self.ref_irr
        noise = 1.0 + self._rng.normal(0.0, c.measurement_noise_std)
        self._measured_w = max(0.0, self._power_w * noise)

    # -- EnergySource protocol ------------------------------------------------
    def measured_power_w(self) -> float:
        return self._measured_w

    # -- simulator-only ground truth ------------------------------------------
    def true_power_w(self) -> float:
        return self._power_w

    def harvested_energy_j(self) -> float:
        return self._power_w * self.dt

    @property
    def daily_clearness(self) -> float:
        """Today's mean irradiance relative to the sunniest day of the trace
        (a stand-in for the clearness index of the stochastic model)."""
        day = int((self._time_s + self.window.offset_s) // DAY_S)
        if not 0 <= day < len(self._daily_irr) or self._daily_irr_max <= 0:
            return float("nan")
        return float(self._daily_irr[day] / self._daily_irr_max)


# ---------------------------------------------------------------------------
# Field backend
# ---------------------------------------------------------------------------
class TraceFieldEnvironment:
    """``FieldEnvironment`` twin replaying recorded soil / atmosphere series.

    * soil moisture: volumetric content [m³/m³] divided by ``field_capacity``
      → fraction of field capacity in [0, ``max_moisture``] (the unit of the
      whole simulator); already-normalised series use ``field_capacity=1``.
    * temperature [°C], relative humidity [% or fraction] replayed as they are.
    * ``event_occurred`` / ``rain_events``: a step change of the *replayed*
      moisture above ``event_change_threshold`` or a zone crossing, exactly as
      in the stochastic model; rain events (onsets of a rain spell, i.e. a
      first interval with more than ``rain_threshold_mm``) are counted from the
      precipitation column when present.
    * irrigation is whatever the recorded field received (not modelled).
    """

    def __init__(
        self,
        trace: Trace,
        cfg: AgricultureConfig,
        timestep_s: float,
        window: TraceWindow,
        *,
        moisture_column: str = "soil_moisture_0_to_7cm",
        temperature_column: str | None = "temperature_2m",
        humidity_column: str | None = "relative_humidity_2m",
        precipitation_column: str | None = "precipitation",
        field_capacity: float = 0.40,
        rain_threshold_mm: float = 0.2,
    ):
        if moisture_column not in trace:
            raise KeyError(f"trace has no column {moisture_column!r}")
        self.trace = trace
        self.cfg = cfg
        self.dt = timestep_s
        self.window = window
        self.moisture_column = moisture_column
        self.temperature_column = temperature_column if temperature_column in trace else None
        self.humidity_column = humidity_column if humidity_column in trace else None
        self.precipitation_column = precipitation_column if precipitation_column in trace else None
        if field_capacity <= 0:
            raise ValueError("field_capacity must be positive")
        self.field_capacity = field_capacity
        self.rain_threshold_mm = rain_threshold_mm
        self._rng = np.random.default_rng()
        self.reset(self._rng, start_time_s=0.0)

    # -- lifecycle --------------------------------------------------------------
    def reset(self, rng: np.random.Generator, start_time_s: float) -> None:
        self._rng = rng
        t = start_time_s + self.window.offset_s
        self.moisture = self._moisture_at(t)
        self.temperature = self._temperature_at(t)
        self.humidity = self._humidity_at(t)
        self.last_change = 0.0
        self.event_occurred = False
        self.rain_events = 0
        self.irrigation_events = 0
        self.total_rain = 0.0
        self.total_irrigation = 0.0
        self._raining = False

    # -- helpers ----------------------------------------------------------------
    def _moisture_at(self, trace_t: float) -> float:
        v = self.trace.value_at(self.moisture_column, trace_t) / self.field_capacity
        return float(np.clip(v, 0.0, self.cfg.max_moisture))

    def _temperature_at(self, trace_t: float) -> float:
        if self.temperature_column is None:
            return self.cfg.temp_mean_c
        return float(self.trace.value_at(self.temperature_column, trace_t))

    def _humidity_at(self, trace_t: float) -> float:
        if self.humidity_column is None:
            return self.cfg.humidity_mean
        h = float(self.trace.value_at(self.humidity_column, trace_t))
        if h > 1.5:  # percent → fraction
            h /= 100.0
        return float(np.clip(h, 0.0, 1.0))

    def zone(self, moisture: float | None = None) -> int:
        m = self.moisture if moisture is None else moisture
        if m < self.cfg.critical_threshold:
            return 2
        if m < self.cfg.warning_threshold:
            return 1
        return 0

    # -- dynamics ---------------------------------------------------------------
    def step(self, time_s: float) -> FieldState:
        """Advance to ``time_s + dt`` by reading the trace."""
        prev = self.moisture
        prev_zone = self.zone(prev)
        t1 = time_s + self.dt + self.window.offset_s
        self.moisture = self._moisture_at(t1)
        self.temperature = self._temperature_at(t1)
        self.humidity = self._humidity_at(t1)
        self.last_change = self.moisture - prev
        if self.precipitation_column is not None:
            rain_mm = self.trace.interval_sum(self.precipitation_column, t1 - self.dt, t1)
            raining = rain_mm > self.rain_threshold_mm
            if raining:
                self.total_rain += rain_mm
                if not self._raining:  # count the onset of a rain spell once
                    self.rain_events += 1
            self._raining = raining
        crossed = self.zone(self.moisture) != prev_zone
        self.event_occurred = abs(self.last_change) >= self.cfg.event_change_threshold or crossed
        return self.state()

    def state(self) -> FieldState:
        return FieldState(
            soil_moisture=self.moisture,
            air_temperature_c=self.temperature,
            relative_humidity=self.humidity,
            last_step_change=self.last_change,
            event_occurred=self.event_occurred,
            zone=self.zone(),
            rain_events=self.rain_events,
            irrigation_events=self.irrigation_events,
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class TraceDrivenEnv(EdgeEngineAwareEnv):
    """EdgeEngine AWARE environment replaying a recorded trace.

    Parameters
    ----------
    trace:
        the recording (see :class:`Trace`).
    config:
        the usual configuration; ``time.episode_days`` sets the window length.
    start_day:
        first day of the trace to replay; ``None`` draws a random admissible
        day at every reset (``reset(options={"start_day": k})`` overrides it
        for one episode).
    harvesting / field:
        which backends to replace (both by default). Keeping the stochastic
        field with a recorded irradiance is useful to isolate the weather
        effect, and vice versa.
    field_capacity:
        volumetric soil moisture corresponding to "1.0" in the simulator.
    source_kwargs / field_kwargs:
        forwarded to the backends (column names, reference irradiance, ...).
    """

    def __init__(
        self,
        trace: Trace,
        config: EdgeEngineAwareConfig | None = None,
        *,
        start_day: int | None = None,
        harvesting: bool = True,
        field: bool = True,
        field_capacity: float = 0.40,
        source_kwargs: Mapping[str, Any] | None = None,
        field_kwargs: Mapping[str, Any] | None = None,
        render_mode: str | None = None,
    ):
        if not (harvesting or field):
            raise ValueError("at least one of harvesting/field must be trace-driven")
        self.trace = trace
        self.start_day = start_day
        self.use_trace_harvesting = harvesting
        self.use_trace_field = field
        self.field_capacity = field_capacity
        self.source_kwargs = dict(source_kwargs or {})
        self.field_kwargs = dict(field_kwargs or {})
        cfg = config if config is not None else EdgeEngineAwareConfig()
        self.window = TraceWindow(trace, cfg.time.episode_days * DAY_S + cfg.time.start_hour * HOUR_S, start_day)
        super().__init__(cfg, render_mode=render_mode)

    def _build_subsystems(self, cfg: EdgeEngineAwareConfig) -> None:
        super()._build_subsystems(cfg)
        dt = cfg.time.timestep_s
        if self.use_trace_harvesting:
            self.source = TraceSolarEnergySource(self.trace, cfg.harvesting, dt, self.window, **self.source_kwargs)
        if self.use_trace_field:
            self.field = TraceFieldEnvironment(
                self.trace, cfg.agriculture, dt, self.window, field_capacity=self.field_capacity, **self.field_kwargs
            )
            # the sensor's ground-truth callable resolves ``self.field`` lazily

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            super(EdgeEngineAwareEnv, self).reset(seed=seed)  # seed np_random before drawing the window
        requested = None if options is None else options.get("start_day")
        self.window.choose(self.np_random, requested)
        obs, info = super().reset(seed=seed, options=options)
        info["trace_start_day"] = self.window.current_day
        return obs, info

    def _make_info(self, reward_components, utility_breakdown, rejected):
        info = super()._make_info(reward_components, utility_breakdown, rejected)
        info["trace_start_day"] = self.window.current_day
        info["trace_day"] = self.window.current_day + self.clock.day_index()
        return info


__all__ = [
    "INSTANT",
    "PRECEDING_MEAN",
    "DEFAULT_KINDS",
    "Trace",
    "TraceColumn",
    "TraceWindow",
    "TraceSolarEnergySource",
    "TraceFieldEnvironment",
    "TraceDrivenEnv",
]
